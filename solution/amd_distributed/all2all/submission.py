import torch
import torch.distributed as dist
from task import input_t, output_t


# ---------------- All2All pytorch impl ----------------
class PyTorchAllToAll:
    META_DIM = 5  # global_exp, src_rank, src_token, src_k, pad

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        # num experts per rank
        self.num_local_experts = cfg.num_experts // world_size
        # max recv tokens per rank
        self.max_recv = cfg.max_num_tokens * world_size

    # ---------- dispatch ----------
    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        device = dp_x.device
        cfg = self.cfg

        # ---------1. get counts of send and recv for each rank -----------
        # 1.1 token nums to send to each rank
        send_counts = [0] * self.world_size
        # 1.2 token id to send to each rank
        token_map = [[] for _ in range(self.world_size)]
        # 1.3 token meta data, need update for combine
        meta_map = [[] for _ in range(self.world_size)]
        for t, expert_list in enumerate(indices.tolist()):
            for k, e in enumerate(expert_list):
                dst_rank = e // self.num_local_experts
                send_counts[dst_rank] += 1
                token_map[dst_rank].append(t)
                meta_map[dst_rank].extend(
                    [e, self.rank, t, k, 0]
                )  # srcGobalExpert, srcRank, srcIndex, expert index

        send_counts_t = torch.tensor(send_counts, dtype=torch.long, device=device)
        # 1.3 token nums to recv from each rank
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        # ---------2. send and recv buffer, order by tokens on each rank ----------
        send_buf = torch.cat([dp_x[idx_list] for idx_list in token_map], dim=0)
        total_recv = int(recv_counts_t.sum().item())
        recv_buf = torch.empty(
            total_recv, cfg.hidden_dim, dtype=cfg.in_dtype, device=device
        )

        # 2.1 meta buf for send and recv
        send_meta = torch.tensor(
            [v for sub in meta_map for v in sub], dtype=torch.int32, device=device
        ).view(-1, self.META_DIM)
        recv_meta = torch.empty(
            total_recv, self.META_DIM, dtype=torch.int32, device=device
        )
        # ---------3. dispatch send_buf to recv_buf by recv and send counts--------------
        dist.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=recv_counts_t.tolist(),
            input_split_sizes=send_counts_t.tolist(),
        )

        dist.all_to_all_single(
            recv_meta.view(-1),
            send_meta.view(-1),
            output_split_sizes=[c * self.META_DIM for c in recv_counts_t.tolist()],
            input_split_sizes=[c * self.META_DIM for c in send_counts_t.tolist()],
        )
        recv_meta = recv_meta.view(-1, self.META_DIM)
        # ---------4. define output tensor of dispatch ------------
        # 4.1 num tokens per expert
        expert_num_tokens = torch.zeros(
            self.num_local_experts, dtype=torch.int32, device=device
        )
        # 4.2 token tensor on each expert
        expert_x = torch.empty(
            (self.num_local_experts, self.max_recv, cfg.hidden_dim),
            dtype=cfg.in_dtype,
            device=device,
        )
        expert_meta = torch.empty(
            (self.num_local_experts, self.max_recv, self.META_DIM),
            dtype=torch.int32,
            device=device,
        )
        # ---------5. dispatch send_meta to recv_meta by recv and send counts------
        # ---------6. write tokens to each expert on each rank ------
        # 6.1 fetch the local expert id of corresponding token i
        for i in range(total_recv):
            global_eid = int(recv_meta[i, 0].item())
            local_eid = global_eid % self.num_local_experts
            # output, store token buf and token meta and token nums of each expert
            expert_x[local_eid, expert_num_tokens[local_eid]] = recv_buf[i]
            expert_meta[local_eid, expert_num_tokens[local_eid]] = recv_meta[i]
            expert_num_tokens[local_eid] += 1
        # 6.2 after dispatch, token nums and token and meta of token on expert
        return expert_num_tokens, expert_x, expert_meta

    # ---------- combine ----------
    def combine(
        self,
        out_tokens: torch.Tensor,  # output, (max num tokens, token dim)
        weights: torch.Tensor,  # topk weight
        expert_meta: torch.Tensor,  # input
        expert_y: torch.Tensor,  # input, (num_local_experts, max_num_tokens * num_dp, token_dim)
        expert_num_tokens: torch.Tensor,
    ):  # input
        device = out_tokens.device
        cfg = self.cfg

        # 1. count send-back tokens in cur rank
        send_counts = [0] * self.world_size
        # 1.1 token that will send back
        y_map = [[] for _ in range(self.world_size)]
        # 1.2 meta info of each token that send back to its src rank
        meta_map = [[] for _ in range(self.world_size)]

        # 2. traverse each token of each local expert of each rank, fill into send_counts and y_map and meta_map
        for local_eid in range(self.num_local_experts):
            cnt = int(expert_num_tokens[local_eid].item())
            for j in range(cnt):
                # meta info token j of local eid
                meta = expert_meta[local_eid, j]
                dst_rank = int(meta[1].item())
                send_counts[dst_rank] += 1
                # token j and its meta that send back to dst rank/local eid
                y_map[dst_rank].append(expert_y[local_eid, j].unsqueeze(0))
                meta_map[dst_rank].extend(meta.tolist())
        # token nums that cur rank plan to send to other ranks
        send_counts_t = torch.tensor(send_counts, dtype=torch.long, device=device)
        # token nums that will recv from other ranks
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        # call all2all to send send counts and recv recv_counts_t at each rank by all2all
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        # 3.send buffers of each rank, that is, the tokens at its experts
        y_map_tensors = []
        for sub_list in y_map:
            if sub_list:
                y_map_tensors.append(torch.cat(sub_list, dim=0))
            else:
                y_map_tensors.append(
                    torch.empty((0, cfg.hidden_dim), dtype=cfg.out_dtype, device=device)
                )
        send_buf = torch.cat(y_map_tensors, dim=0)
        # 4. flatten send meta by tokens
        send_meta = torch.tensor(
            [v for sub in meta_map for v in sub], dtype=torch.int32, device=device
        ).view(-1, self.META_DIM)
        # 5. total recv tokens of cur rank
        total_recv = int(recv_counts_t.sum().item())
        # 6. recv buffer of cur rank
        recv_buf = torch.empty(
            total_recv, cfg.hidden_dim, dtype=cfg.out_dtype, device=device
        )
        recv_meta = torch.empty(
            total_recv, self.META_DIM, dtype=torch.int32, device=device
        )
        # 7. call all2all to send and recv for each rank
        dist.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=recv_counts_t.tolist(),
            input_split_sizes=send_counts_t.tolist(),
        )
        # 8. call all2all to send meta and recv meta for each rank
        dist.all_to_all_single(
            recv_meta.view(-1),
            send_meta.view(-1),
            output_split_sizes=[c * self.META_DIM for c in recv_counts_t.tolist()],
            input_split_sizes=[c * self.META_DIM for c in send_counts_t.tolist()],
        )
        # 9. restore recv meta
        recv_meta = recv_meta.view(-1, self.META_DIM)

        # 10. write back tokens from recv buf, per meta info, and do weighted sum
        for i in range(total_recv):
            src_token = int(recv_meta[i, 2].item())
            src_k = int(recv_meta[i, 3].item())
            src_rank = int(recv_meta[i, 1].item())
            w = weights[src_token, src_k].to(torch.float32)
            out_tokens[src_token] += recv_buf[i].to(torch.float32) * w

        return out_tokens

def manual_all_to_all_single(recv_buf, send_buf, output_split_sizes, input_split_sizes, group=None):
    """
    使用 send/recv 手动实现 all_to_all_single 的功能。
    注意：此实现假设所有 rank 同时调用，并且 split_sizes 长度等于 world_size。
    """
    if group is not None:
        # 注意：send/recv 的 dst/src 是全局 rank。如果使用 group，需要转换。
        # 为简化，这里假设 group 是默认组或处理了 rank 转换。
        # 更严格的实现需要 dist.get_global_rank(group, group_rank)
        raise NotImplementedError("This manual implementation does not handle custom groups directly.")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        # 单进程情况，直接复制
        if recv_buf is not send_buf: # 避免不必要的拷贝
            recv_buf.copy_(send_buf)
        return

    # 1. 根据 input_split_sizes 切分 send_buf
    send_slices = []
    start_idx = 0
    for i, size in enumerate(input_split_sizes):
        end_idx = start_idx + size
        if size > 0:
            send_slices.append((i, send_buf[start_idx:end_idx])) # (目标 rank, 数据)
        else:
            # 发送空张量
            send_slices.append((i, send_buf[0:0])) # 创建一个空张量，形状正确
        start_idx = end_idx

    # 2. 准备接收缓冲区的视图 (如果 recv_buf 不是空的)
    recv_views = []
    start_idx = 0
    for size in output_split_sizes:
        end_idx = start_idx + size
        if size > 0:
             recv_views.append(recv_buf[start_idx:end_idx])
        else:
             recv_views.append(recv_buf[0:0]) # 空视图
        start_idx = end_idx

    # 3. 执行点对点通信
    requests = []
    # 遵循避免死锁的顺序：先处理 rank < current_rank 的，再处理 rank > current_rank 的
    # 处理 rank < current_rank: 先 recv 再 send
    for src_rank in range(rank):
        if output_split_sizes[src_rank] > 0: # 只有当预期接收数据时才 recv
            req_recv = dist.irecv(tensor=recv_views[src_rank], src=src_rank)
            requests.append(req_recv)
        if input_split_sizes[src_rank] > 0: # 只有当有数据要发送时才 send
             # send_slices[src_rank][0] 是目标 rank，应该等于 src_rank
            req_send = dist.isend(tensor=send_slices[src_rank][1], dst=src_rank)
            requests.append(req_send)

    # 处理自己 (rank == current_rank): 可以直接复制或发送给自己
    # 为了避免潜在问题，我们也可以用 send/recv，但通常直接复制更快
    if input_split_sizes[rank] > 0 and output_split_sizes[rank] > 0:
        # 确保是同一块内存区域的复制
        if not recv_views[rank].data_ptr() == send_slices[rank][1].data_ptr():
            recv_views[rank].copy_(send_slices[rank][1])

    # 处理 rank > current_rank: 先 send 再 recv
    for dst_rank in range(rank + 1, world_size):
        if input_split_sizes[dst_rank] > 0: # 先 send
            req_send = dist.isend(tensor=send_slices[dst_rank][1], dst=dst_rank)
            requests.append(req_send)
        if output_split_sizes[dst_rank] > 0: # 再 recv
            req_recv = dist.irecv(tensor=recv_views[dst_rank], src=dst_rank)
            requests.append(req_recv)

    # 4. 等待所有异步操作完成
    for req in requests:
        req.wait()

# ---------------- 使用手动 all_to_all 的 All2All 实现 ----------------
class OptimizedByQwenPyTorchAllToAll:
    META_DIM = 5

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size
        self.max_recv = cfg.max_num_tokens * world_size
        # 不再创建额外的 group

    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        device = dp_x.device
        cfg = self.cfg

        # --- 与之前相同的逻辑准备数据 ---
        send_counts = [0] * self.world_size
        token_map = [[] for _ in range(self.world_size)]
        meta_map = [[] for _ in range(self.world_size)]

        for t, expert_list in enumerate(indices.tolist()):
            for k, e in enumerate(expert_list):
                dst_rank = e // self.num_local_experts
                send_counts[dst_rank] += 1
                token_map[dst_rank].append(t)
                meta_map[dst_rank].extend([e, self.rank, t, k, 0])

        send_counts_t = torch.tensor(send_counts, dtype=torch.long, device=device)
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        # 使用手动实现的 all_to_all_single
        manual_all_to_all_single(recv_counts_t, send_counts_t, [1]*self.world_size, [1]*self.world_size)

        send_buf_list = []
        for idx_list in token_map:
            if idx_list:
                send_buf_list.append(dp_x[idx_list])
            else:
                send_buf_list.append(torch.empty((0, cfg.hidden_dim), dtype=cfg.in_dtype, device=device))
        send_buf = torch.cat(send_buf_list, dim=0) if send_buf_list else \
                  torch.empty((0, cfg.hidden_dim), dtype=cfg.in_dtype, device=device)

        flat_meta = [v for sub in meta_map for v in sub]
        send_meta = torch.tensor(flat_meta, dtype=torch.int32, device=device).view(-1, self.META_DIM) if flat_meta else \
                   torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)

        total_recv = int(recv_counts_t.sum().item())
        recv_split_sizes = recv_counts_t.tolist()
        send_split_sizes = send_counts_t.tolist()

        recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)
        recv_meta = torch.empty(total_recv, self.META_DIM, dtype=torch.int32, device=device)

        # --- 使用手动实现进行数据通信 ---
        manual_all_to_all_single(
            recv_buf,
            send_buf,
            recv_split_sizes,
            send_split_sizes,
        )

        manual_all_to_all_single(
            recv_meta.view(-1),
            send_meta.view(-1),
            [c * self.META_DIM for c in recv_split_sizes],
            [c * self.META_DIM for c in send_split_sizes],
        )
        recv_meta = recv_meta.view(-1, self.META_DIM)

        # --- 与之前相同的本地处理逻辑 ---
        expert_num_tokens = torch.zeros(self.num_local_experts, dtype=torch.int32, device=device)
        expert_x = torch.empty((self.num_local_experts, self.max_recv, cfg.hidden_dim),
                               dtype=cfg.in_dtype, device=device)
        expert_meta = torch.empty((self.num_local_experts, self.max_recv, self.META_DIM),
                                  dtype=torch.int32, device=device)

        if total_recv > 0:
            global_eids = recv_meta[:, 0].to(torch.long)
            local_eids = global_eids % self.num_local_experts
            for i in range(total_recv):
                local_eid = int(local_eids[i].item())
                pos = int(expert_num_tokens[local_eid].item())
                expert_x[local_eid, pos] = recv_buf[i]
                expert_meta[local_eid, pos] = recv_meta[i]
                expert_num_tokens[local_eid] += 1

        return expert_num_tokens, expert_x, expert_meta

    def combine(self, out_tokens: torch.Tensor, weights: torch.Tensor,
                expert_meta: torch.Tensor, expert_y: torch.Tensor,
                expert_num_tokens: torch.Tensor):
        device = out_tokens.device
        cfg = self.cfg

        # --- 与之前相同的逻辑准备数据 ---
        send_counts = [0] * self.world_size
        y_map = [[] for _ in range(self.world_size)]
        meta_map = [[] for _ in range(self.world_size)]

        for local_eid in range(self.num_local_experts):
            cnt = int(expert_num_tokens[local_eid].item())
            for j in range(cnt):
                meta = expert_meta[local_eid, j]
                dst_rank = int(meta[1].item())
                send_counts[dst_rank] += 1
                y_map[dst_rank].append(expert_y[local_eid, j])
                meta_map[dst_rank].extend(meta.tolist())

        send_counts_t = torch.tensor(send_counts, dtype=torch.long, device=device)
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        manual_all_to_all_single(recv_counts_t, send_counts_t, [1]*self.world_size, [1]*self.world_size)

        y_map_tensors = []
        for sub_list in y_map:
            if sub_list:
                y_map_tensors.append(torch.stack(sub_list, dim=0))
            else:
                y_map_tensors.append(torch.empty((0, cfg.hidden_dim), dtype=cfg.out_dtype, device=device))
        send_buf = torch.cat(y_map_tensors, dim=0) if y_map_tensors else \
                  torch.empty((0, cfg.hidden_dim), dtype=cfg.out_dtype, device=device)

        flat_meta = [v for sub in meta_map for v in sub]
        send_meta = torch.tensor(flat_meta, dtype=torch.int32, device=device).view(-1, self.META_DIM) if flat_meta else \
                   torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)

        total_recv = int(recv_counts_t.sum().item())
        recv_split_sizes = recv_counts_t.tolist()
        send_split_sizes = send_counts_t.tolist()

        recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
        recv_meta = torch.empty(total_recv, self.META_DIM, dtype=torch.int32, device=device)

        # --- 使用手动实现进行数据通信 ---
        manual_all_to_all_single(
            recv_buf,
            send_buf,
            recv_split_sizes,
            send_split_sizes,
        )

        manual_all_to_all_single(
            recv_meta.view(-1),
            send_meta.view(-1),
            [c * self.META_DIM for c in recv_split_sizes],
            [c * self.META_DIM for c in send_split_sizes],
        )
        recv_meta = recv_meta.view(-1, self.META_DIM)

        # --- 与之前相同的聚合逻辑 ---
        if total_recv > 0:
            src_tokens = recv_meta[:, 2].to(torch.long)
            src_ks = recv_meta[:, 3].to(torch.long)
            weights_selected = weights[src_tokens, src_ks]
            weighted_values = recv_buf * weights_selected.unsqueeze(-1).to(recv_buf.dtype)
            weighted_values = weighted_values.to(out_tokens.dtype)
            src_tokens_expanded = src_tokens.unsqueeze(-1).expand_as(weighted_values)
            out_tokens.scatter_add_(0, src_tokens_expanded, weighted_values)

        return out_tokens

class OptimizedByDpskPyTorchAllToAll:
    META_DIM = 5  # global_exp, src_rank, src_token, src_k, pad

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size
        self.max_recv = cfg.max_num_tokens * world_size

        # 根据拓扑结构分组
        self.group0 = list(range(0, 4))  # GPU0-3
        self.group1 = list(range(4, 8))  # GPU4-7
        self.is_group0 = self.rank in self.group0
        
        # 创建组内通信子
        if self.is_group0:
            intra_group_ranks = self.group0
        else:
            intra_group_ranks = self.group1
            
        self.intra_group = dist.new_group(intra_group_ranks)
        self.intra_group_size = len(intra_group_ranks)
        self.intra_group_rank = intra_group_ranks.index(self.rank)

    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        device = dp_x.device
        cfg = self.cfg

        # 第一步：组内all-to-all通信
        intra_send_counts = torch.zeros(self.intra_group_size, dtype=torch.long, device=device)
        token_map_intra = [[] for _ in range(self.intra_group_size)]
        meta_map_intra = [[] for _ in range(self.intra_group_size)]
        
        # 收集组内发送数据
        for t, expert_list in enumerate(indices.tolist()):
            for k, e in enumerate(expert_list):
                dst_rank = e // self.num_local_experts
                
                # 如果目标rank在同一组内
                if (self.is_group0 and dst_rank in self.group0) or \
                   (not self.is_group0 and dst_rank in self.group1):
                    intra_dst_rank = self.group0.index(dst_rank) if self.is_group0 else self.group1.index(dst_rank)
                    intra_send_counts[intra_dst_rank] += 1
                    token_map_intra[intra_dst_rank].append(t)
                    meta_map_intra[intra_dst_rank].extend([e, self.rank, t, k, 0])

        # 组内all-to-all通信
        intra_recv_counts = torch.zeros(self.intra_group_size, dtype=torch.long, device=device)
        dist.all_to_all_single(intra_recv_counts, intra_send_counts, group=self.intra_group)
        
        # 准备组内发送缓冲区
        intra_send_buf_list = []
        for idx_list in token_map_intra:
            if idx_list:
                intra_send_buf_list.append(dp_x[idx_list])
            else:
                intra_send_buf_list.append(torch.empty((0, cfg.hidden_dim), dtype=cfg.in_dtype, device=device))
        
        intra_send_buf = torch.cat(intra_send_buf_list, dim=0) if intra_send_buf_list else \
                        torch.empty((0, cfg.hidden_dim), dtype=cfg.in_dtype, device=device)
        
        # 准备组内发送meta数据
        flat_meta_intra = [v for sub in meta_map_intra for v in sub]
        intra_send_meta = torch.tensor(flat_meta_intra, dtype=torch.int32, device=device).view(-1, self.META_DIM) if flat_meta_intra else \
                         torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)

        # 组内接收缓冲区
        intra_total_recv = int(intra_recv_counts.sum().item())
        intra_recv_buf = torch.empty(intra_total_recv, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)
        intra_recv_meta = torch.empty(intra_total_recv, self.META_DIM, dtype=torch.int32, device=device)

        # 执行组内通信
        dist.all_to_all_single(
            intra_recv_buf,
            intra_send_buf,
            output_split_sizes=intra_recv_counts.tolist(),
            input_split_sizes=intra_send_counts.tolist(),
            group=self.intra_group
        )

        dist.all_to_all_single(
            intra_recv_meta.view(-1),
            intra_send_meta.view(-1),
            output_split_sizes=[c * self.META_DIM for c in intra_recv_counts.tolist()],
            input_split_sizes=[c * self.META_DIM for c in intra_send_counts.tolist()],
            group=self.intra_group
        )
        intra_recv_meta = intra_recv_meta.view(-1, self.META_DIM)

        # 第二步：处理跨组通信
        inter_send_counts = torch.zeros(self.world_size - self.intra_group_size, dtype=torch.long, device=device)
        token_map_inter = [[] for _ in range(self.world_size - self.intra_group_size)]
        meta_map_inter = [[] for _ in range(self.world_size - self.intra_group_size)]
        
        # 收集跨组发送数据
        for t, expert_list in enumerate(indices.tolist()):
            for k, e in enumerate(expert_list):
                dst_rank = e // self.num_local_experts
                
                # 如果目标rank在另一组
                if (self.is_group0 and dst_rank in self.group1) or \
                   (not self.is_group0 and dst_rank in self.group0):
                    inter_dst_idx = dst_rank - 4 if self.is_group0 else dst_rank
                    inter_send_counts[inter_dst_idx] += 1
                    token_map_inter[inter_dst_idx].append(t)
                    meta_map_inter[inter_dst_idx].extend([e, self.rank, t, k, 0])

        # 全局跨组通信
        inter_recv_counts = torch.zeros(self.world_size - self.intra_group_size, dtype=torch.long, device=device)
        dist.all_to_all_single(inter_recv_counts, inter_send_counts)
        
        # 准备跨组发送缓冲区
        inter_send_buf_list = []
        for idx_list in token_map_inter:
            if idx_list:
                inter_send_buf_list.append(dp_x[idx_list])
            else:
                inter_send_buf_list.append(torch.empty((0, cfg.hidden_dim), dtype=cfg.in_dtype, device=device))
        
        inter_send_buf = torch.cat(inter_send_buf_list, dim=0) if inter_send_buf_list else \
                        torch.empty((0, cfg.hidden_dim), dtype=cfg.in_dtype, device=device)
        
        # 准备跨组发送meta数据
        flat_meta_inter = [v for sub in meta_map_inter for v in sub]
        inter_send_meta = torch.tensor(flat_meta_inter, dtype=torch.int32, device=device).view(-1, self.META_DIM) if flat_meta_inter else \
                         torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)

        # 跨组接收缓冲区
        inter_total_recv = int(inter_recv_counts.sum().item())
        inter_recv_buf = torch.empty(inter_total_recv, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)
        inter_recv_meta = torch.empty(inter_total_recv, self.META_DIM, dtype=torch.int32, device=device)

        # 执行跨组通信
        dist.all_to_all_single(
            inter_recv_buf,
            inter_send_buf,
            output_split_sizes=inter_recv_counts.tolist(),
            input_split_sizes=inter_send_counts.tolist(),
        )

        dist.all_to_all_single(
            inter_recv_meta.view(-1),
            inter_send_meta.view(-1),
            output_split_sizes=[c * self.META_DIM for c in inter_recv_counts.tolist()],
            input_split_sizes=[c * self.META_DIM for c in inter_send_counts.tolist()],
        )
        inter_recv_meta = inter_recv_meta.view(-1, self.META_DIM)

        # 合并组内和跨组接收结果
        total_recv = intra_total_recv + inter_total_recv
        recv_buf = torch.cat([intra_recv_buf, inter_recv_buf], dim=0)
        recv_meta = torch.cat([intra_recv_meta, inter_recv_meta], dim=0)

        # 分发到本地专家
        expert_num_tokens = torch.zeros(self.num_local_experts, dtype=torch.int32, device=device)
        expert_x = torch.empty((self.num_local_experts, self.max_recv, cfg.hidden_dim),
                               dtype=cfg.in_dtype, device=device)
        expert_meta = torch.empty((self.num_local_experts, self.max_recv, self.META_DIM),
                                  dtype=torch.int32, device=device)

        if total_recv > 0:
            global_eids = recv_meta[:, 0].to(torch.long)
            local_eids = global_eids % self.num_local_experts
            
            for i in range(total_recv):
                local_eid = int(local_eids[i].item())
                pos = int(expert_num_tokens[local_eid].item())
                expert_x[local_eid, pos] = recv_buf[i]
                expert_meta[local_eid, pos] = recv_meta[i]
                expert_num_tokens[local_eid] += 1

        return expert_num_tokens, expert_x, expert_meta

    def combine(self, out_tokens: torch.Tensor, weights: torch.Tensor,
                expert_meta: torch.Tensor, expert_y: torch.Tensor,
                expert_num_tokens: torch.Tensor):
        device = out_tokens.device
        cfg = self.cfg

        # 收集所有要发送回的数据
        intra_send_counts = torch.zeros(self.intra_group_size, dtype=torch.long, device=device)
        intra_y_map = [[] for _ in range(self.intra_group_size)]
        intra_meta_map = [[] for _ in range(self.intra_group_size)]
        
        inter_send_counts = torch.zeros(self.world_size - self.intra_group_size, dtype=torch.long, device=device)
        inter_y_map = [[] for _ in range(self.world_size - self.intra_group_size)]
        inter_meta_map = [[] for _ in range(self.world_size - self.intra_group_size)]

        for local_eid in range(self.num_local_experts):
            cnt = int(expert_num_tokens[local_eid].item())
            for j in range(cnt):
                meta = expert_meta[local_eid, j]
                dst_rank = int(meta[1].item())
                
                if (self.is_group0 and dst_rank in self.group0) or \
                   (not self.is_group0 and dst_rank in self.group1):
                    # 组内通信
                    intra_dst_rank = self.group0.index(dst_rank) if self.is_group0 else self.group1.index(dst_rank)
                    intra_send_counts[intra_dst_rank] += 1
                    intra_y_map[intra_dst_rank].append(expert_y[local_eid, j])
                    intra_meta_map[intra_dst_rank].extend(meta.tolist())
                else:
                    # 跨组通信
                    inter_dst_idx = dst_rank - 4 if self.is_group0 else dst_rank
                    inter_send_counts[inter_dst_idx] += 1
                    inter_y_map[inter_dst_idx].append(expert_y[local_eid, j])
                    inter_meta_map[inter_dst_idx].extend(meta.tolist())

        # 组内通信
        intra_recv_counts = torch.zeros(self.intra_group_size, dtype=torch.long, device=device)
        dist.all_to_all_single(intra_recv_counts, intra_send_counts, group=self.intra_group)
        
        # 准备组内发送缓冲区
        intra_send_buf_list = []
        for sub_list in intra_y_map:
            if sub_list:
                intra_send_buf_list.append(torch.stack(sub_list, dim=0))
            else:
                intra_send_buf_list.append(torch.empty((0, cfg.hidden_dim), dtype=cfg.out_dtype, device=device))
        
        intra_send_buf = torch.cat(intra_send_buf_list, dim=0) if intra_send_buf_list else \
                        torch.empty((0, cfg.hidden_dim), dtype=cfg.out_dtype, device=device)
        
        # 准备组内发送meta数据
        flat_meta_intra = [v for sub in intra_meta_map for v in sub]
        intra_send_meta = torch.tensor(flat_meta_intra, dtype=torch.int32, device=device).view(-1, self.META_DIM) if flat_meta_intra else \
                         torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)

        # 组内接收缓冲区
        intra_total_recv = int(intra_recv_counts.sum().item())
        intra_recv_buf = torch.empty(intra_total_recv, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
        intra_recv_meta = torch.empty(intra_total_recv, self.META_DIM, dtype=torch.int32, device=device)

        # 执行组内通信
        dist.all_to_all_single(
            intra_recv_buf,
            intra_send_buf,
            output_split_sizes=intra_recv_counts.tolist(),
            input_split_sizes=intra_send_counts.tolist(),
            group=self.intra_group
        )

        dist.all_to_all_single(
            intra_recv_meta.view(-1),
            intra_send_meta.view(-1),
            output_split_sizes=[c * self.META_DIM for c in intra_recv_counts.tolist()],
            input_split_sizes=[c * self.META_DIM for c in intra_send_counts.tolist()],
            group=self.intra_group
        )
        intra_recv_meta = intra_recv_meta.view(-1, self.META_DIM)

        # 跨组通信
        inter_recv_counts = torch.zeros(self.world_size - self.intra_group_size, dtype=torch.long, device=device)
        dist.all_to_all_single(inter_recv_counts, inter_send_counts)
        
        # 准备跨组发送缓冲区
        inter_send_buf_list = []
        for sub_list in inter_y_map:
            if sub_list:
                inter_send_buf_list.append(torch.stack(sub_list, dim=0))
            else:
                inter_send_buf_list.append(torch.empty((0, cfg.hidden_dim), dtype=cfg.out_dtype, device=device))
        
        inter_send_buf = torch.cat(inter_send_buf_list, dim=0) if inter_send_buf_list else \
                        torch.empty((0, cfg.hidden_dim), dtype=cfg.out_dtype, device=device)
        
        # 准备跨组发送meta数据
        flat_meta_inter = [v for sub in inter_meta_map for v in sub]
        inter_send_meta = torch.tensor(flat_meta_inter, dtype=torch.int32, device=device).view(-1, self.META_DIM) if flat_meta_inter else \
                         torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)

        # 跨组接收缓冲区
        inter_total_recv = int(inter_recv_counts.sum().item())
        inter_recv_buf = torch.empty(inter_total_recv, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
        inter_recv_meta = torch.empty(inter_total_recv, self.META_DIM, dtype=torch.int32, device=device)

        # 执行跨组通信
        dist.all_to_all_single(
            inter_recv_buf,
            inter_send_buf,
            output_split_sizes=inter_recv_counts.tolist(),
            input_split_sizes=inter_send_counts.tolist(),
        )

        dist.all_to_all_single(
            inter_recv_meta.view(-1),
            inter_send_meta.view(-1),
            output_split_sizes=[c * self.META_DIM for c in inter_recv_counts.tolist()],
            input_split_sizes=[c * self.META_DIM for c in inter_send_counts.tolist()],
        )
        inter_recv_meta = inter_recv_meta.view(-1, self.META_DIM)

        # 合并接收结果
        total_recv = intra_total_recv + inter_total_recv
        recv_buf = torch.cat([intra_recv_buf, inter_recv_buf], dim=0)
        recv_meta = torch.cat([intra_recv_meta, inter_recv_meta], dim=0)

        # 处理接收到的数据
        if total_recv > 0:
            src_tokens = recv_meta[:, 2].to(torch.long)
            src_ks = recv_meta[:, 3].to(torch.long)
            
            weights_selected = weights[src_tokens, src_ks]
            weighted_values = recv_buf * weights_selected.unsqueeze(-1).to(recv_buf.dtype)
            
            weighted_values = weighted_values.to(out_tokens.dtype)
            src_tokens_expanded = src_tokens.unsqueeze(-1).expand_as(weighted_values)
            
            out_tokens.scatter_add_(0, src_tokens_expanded, weighted_values)

        return out_tokens

def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)

    # ata = PyTorchAllToAll(cfg, rank, world_size)
    ata = OptimizedByQwenPyTorchAllToAll(cfg, rank, world_size)
    # ata = OptimizedByDpskPyTorchAllToAll(cfg, rank, world_size)

    expert_num, expert_x, expert_meta = ata.dispatch(rank_data.x, rank_data.indices)
    expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)
    y = torch.zeros(
        cfg.max_num_tokens,
        cfg.hidden_dim,
        dtype=cfg.out_dtype,
        device=rank_data.x.device,
    )

    ata.combine(y, rank_data.weights, expert_meta, expert_y, expert_num)

    return y[: rank_data.num_tokens]
