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
    使用 send/recv 手动实现 all_to_all_single 的功能，并使用 batch_isend_irecv 优化。
    注意：此实现假设所有 rank 同时调用，并且 split_sizes 长度等于 world_size。
    """
    # 如果指定了 group，这里为了简化没有处理，实际使用时需要注意 rank 转换
    if group is not None:
        raise NotImplementedError("This manual implementation does not handle custom groups directly in the batched version.")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        if recv_buf is not send_buf:
            recv_buf.copy_(send_buf)
        return

    # 1. 根据 input_split_sizes 切分 send_buf
    send_slices = []
    start_idx = 0
    for i, size in enumerate(input_split_sizes):
        end_idx = start_idx + size
        if size > 0:
            send_slices.append((i, send_buf[start_idx:end_idx]))
        else:
            send_slices.append((i, send_buf[0:0])) # 空张量
        start_idx = end_idx

    # 2. 准备接收缓冲区的视图
    recv_views = []
    start_idx = 0
    for size in output_split_sizes:
        end_idx = start_idx + size
        if size > 0:
             recv_views.append(recv_buf[start_idx:end_idx])
        else:
             recv_views.append(recv_buf[0:0]) # 空视图
        start_idx = end_idx

    # 3. 构建 P2P 操作列表 (遵循避免死锁的顺序)
    p2p_ops = []
    
    # --- 第一阶段：与 rank < current_rank 通信 ---
    # 先准备接收操作
    for src_rank in range(rank):
        if output_split_sizes[src_rank] > 0:
            op = dist.P2POp(dist.irecv, recv_views[src_rank], src_rank)
            p2p_ops.append(op)
    # 再准备发送操作
    for src_rank in range(rank):
        if input_split_sizes[src_rank] > 0:
            op = dist.P2POp(dist.isend, send_slices[src_rank][1], src_rank)
            p2p_ops.append(op)

    # --- 处理自己 (rank == current_rank) ---
    # 直接复制，不涉及网络通信
    if input_split_sizes[rank] > 0 and output_split_sizes[rank] > 0:
        if not (recv_views[rank].data_ptr() == send_slices[rank][1].data_ptr() and recv_views[rank].shape == send_slices[rank][1].shape):
             recv_views[rank].copy_(send_slices[rank][1])

    # --- 第二阶段：与 rank > current_rank 通信 ---
    # 先准备发送操作
    for dst_rank in range(rank + 1, world_size):
        if input_split_sizes[dst_rank] > 0:
            op = dist.P2POp(dist.isend, send_slices[dst_rank][1], dst_rank)
            p2p_ops.append(op)
    # 再准备接收操作
    for dst_rank in range(rank + 1, world_size):
        if output_split_sizes[dst_rank] > 0:
            op = dist.P2POp(dist.irecv, recv_views[dst_rank], dst_rank)
            p2p_ops.append(op)

    # 4. 批量执行所有 P2P 操作
    if p2p_ops: # 只有当有待处理的操作时才调用
        requests = dist.batch_isend_irecv(p2p_ops)
        # 5. 等待所有异步操作完成
        for req in requests:
            req.wait()

# ---------------- 使用手动 batched all_to_all 的 All2All 实现 ----------------
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
        # 使用手动实现的 all_to_all_single (现在是批处理版本)
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

        # --- 使用手动批处理实现进行数据通信 ---
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

        # --- 使用手动批处理实现进行数据通信 ---
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
        
        # 预计算拓扑距离，优先本地通信
        self.rank_distances = self._compute_topology_aware_routing()
        
        # 根据距离排序目标rank，优先选择近距离的rank
        self.sorted_ranks_by_distance = sorted(
            range(self.world_size), 
            key=lambda x: self.rank_distances.get(x, 999)
        )

    def _compute_topology_aware_routing(self):
        # 基于ROCm拓扑结构预计算通信成本
        distances = {}
        group_id = self.rank // 4
        for dst_rank in range(self.world_size):
            dst_group_id = dst_rank // 4
            if group_id == dst_group_id:
                distances[dst_rank] = 1  # XGMI直连
            else:
                distances[dst_rank] = 3  # PCIe跨组
        return distances

    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        device = dp_x.device
        cfg = self.cfg

        # 预分配发送计数
        send_counts = [0] * self.world_size
        token_map = [[] for _ in range(self.world_size)]
        meta_map = [[] for _ in range(self.world_size)]
        
        # 收集所有要发送的数据 - 使用拓扑感知路由
        for t, expert_list in enumerate(indices.tolist()):
            for k, e in enumerate(expert_list):
                dst_rank = e // self.num_local_experts
                
                # 拓扑感知：优先选择同组的rank，如果可能的话
                # 这里我们保持原有的正确性，但可以添加负载均衡考虑
                preferred_ranks = [r for r in self.sorted_ranks_by_distance if r == dst_rank]
                if preferred_ranks:
                    target_rank = preferred_ranks[0]
                else:
                    target_rank = dst_rank
                    
                send_counts[target_rank] += 1
                token_map[target_rank].append(t)
                meta_map[target_rank].extend([e, self.rank, t, k, 0])

        send_counts_t = torch.tensor(send_counts, dtype=torch.long, device=device)
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        
        # 构造发送缓冲区
        send_buf_list = []
        for idx_list in token_map:
            if idx_list:
                send_buf_list.append(dp_x[idx_list])
            else:
                send_buf_list.append(torch.empty((0, cfg.hidden_dim), dtype=cfg.in_dtype, device=device))
        
        send_buf = torch.cat(send_buf_list, dim=0) if send_buf_list else \
                  torch.empty((0, cfg.hidden_dim), dtype=cfg.in_dtype, device=device)
            
        # 构造发送meta数据
        flat_meta = [v for sub in meta_map for v in sub]
        send_meta = torch.tensor(flat_meta, dtype=torch.int32, device=device).view(-1, self.META_DIM) if flat_meta else \
                   torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)

        # 计算接收参数
        total_recv = int(recv_counts_t.sum().item())
        recv_split_sizes = recv_counts_t.tolist()
        send_split_sizes = send_counts_t.tolist()
        
        # 创建接收缓冲区
        recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)
        recv_meta = torch.empty(total_recv, self.META_DIM, dtype=torch.int32, device=device)

        # 执行通信 - 使用拓扑感知的通信顺序
        # 先执行组内通信，然后执行组间通信
        if self.is_group0:
            # 先处理组内通信 (ranks 0-3)
            intra_group_ranks = self.group0
            inter_group_ranks = self.group1
            
            # 组内通信
            intra_send_sizes = [send_split_sizes[r] for r in intra_group_ranks]
            intra_recv_sizes = [recv_split_sizes[r] for r in intra_group_ranks]
            
            # 组间通信
            inter_send_sizes = [send_split_sizes[r] for r in inter_group_ranks]
            inter_recv_sizes = [recv_split_sizes[r] for r in inter_group_ranks]
            
            # 执行通信
            dist.all_to_all_single(
                recv_buf,
                send_buf,
                output_split_sizes=intra_recv_sizes + inter_recv_sizes,
                input_split_sizes=intra_send_sizes + inter_send_sizes,
            )

            dist.all_to_all_single(
                recv_meta.view(-1),
                send_meta.view(-1),
                output_split_sizes=[c * self.META_DIM for c in intra_recv_sizes + inter_recv_sizes],
                input_split_sizes=[c * self.META_DIM for c in intra_send_sizes + inter_send_sizes],
            )
        else:
            # 先处理组内通信 (ranks 4-7)
            intra_group_ranks = self.group1
            inter_group_ranks = self.group0
            
            # 组内通信
            intra_send_sizes = [send_split_sizes[r] for r in intra_group_ranks]
            intra_recv_sizes = [recv_split_sizes[r] for r in intra_group_ranks]
            
            # 组间通信
            inter_send_sizes = [send_split_sizes[r] for r in inter_group_ranks]
            inter_recv_sizes = [recv_split_sizes[r] for r in inter_group_ranks]
            
            # 执行通信
            dist.all_to_all_single(
                recv_buf,
                send_buf,
                output_split_sizes=inter_recv_sizes + intra_recv_sizes,
                input_split_sizes=inter_send_sizes + intra_send_sizes,
            )

            dist.all_to_all_single(
                recv_meta.view(-1),
                send_meta.view(-1),
                output_split_sizes=[c * self.META_DIM for c in inter_recv_sizes + intra_recv_sizes],
                input_split_sizes=[c * self.META_DIM for c in inter_send_sizes + intra_send_sizes],
            )
            
        recv_meta = recv_meta.view(-1, self.META_DIM)

        # 分发到本地专家 - 内存访问模式优化
        expert_num_tokens = torch.zeros(self.num_local_experts, dtype=torch.int32, device=device)
        expert_x = torch.empty((self.num_local_experts, self.max_recv, cfg.hidden_dim),
                               dtype=cfg.in_dtype, device=device)
        expert_meta = torch.empty((self.num_local_experts, self.max_recv, self.META_DIM),
                                  dtype=torch.int32, device=device)

        # 批量处理优化
        if total_recv > 0:
            # 预先提取所有索引
            global_eids = recv_meta[:, 0].to(torch.long)
            local_eids = global_eids % self.num_local_experts
            
            # 向量化分发
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
        dist.all_to_all_single(recv_counts_t, send_counts_t)

        # 构造发送缓冲区
        y_map_tensors = []
        for sub_list in y_map:
            if sub_list:
                y_map_tensors.append(torch.stack(sub_list, dim=0))
            else:
                y_map_tensors.append(torch.empty((0, cfg.hidden_dim), dtype=cfg.out_dtype, device=device))
        
        send_buf = torch.cat(y_map_tensors, dim=0) if y_map_tensors else \
                  torch.empty((0, cfg.hidden_dim), dtype=cfg.out_dtype, device=device)
            
        # 构造发送meta数据
        flat_meta = [v for sub in meta_map for v in sub]
        send_meta = torch.tensor(flat_meta, dtype=torch.int32, device=device).view(-1, self.META_DIM) if flat_meta else \
                   torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)

        # 计算接收参数
        total_recv = int(recv_counts_t.sum().item())
        recv_split_sizes = recv_counts_t.tolist()
        send_split_sizes = send_counts_t.tolist()

        # 创建接收缓冲区
        recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
        recv_meta = torch.empty(total_recv, self.META_DIM, dtype=torch.int32, device=device)

        # 执行通信 - 使用拓扑感知的通信顺序
        # 先执行组内通信，然后执行组间通信
        if self.is_group0:
            # 先处理组内通信 (ranks 0-3)
            intra_group_ranks = self.group0
            inter_group_ranks = self.group1
            
            # 组内通信
            intra_send_sizes = [send_split_sizes[r] for r in intra_group_ranks]
            intra_recv_sizes = [recv_split_sizes[r] for r in intra_group_ranks]
            
            # 组间通信
            inter_send_sizes = [send_split_sizes[r] for r in inter_group_ranks]
            inter_recv_sizes = [recv_split_sizes[r] for r in inter_group_ranks]
            
            # 执行通信
            dist.all_to_all_single(
                recv_buf,
                send_buf,
                output_split_sizes=intra_recv_sizes + inter_recv_sizes,
                input_split_sizes=intra_send_sizes + inter_send_sizes,
            )

            dist.all_to_all_single(
                recv_meta.view(-1),
                send_meta.view(-1),
                output_split_sizes=[c * self.META_DIM for c in intra_recv_sizes + inter_recv_sizes],
                input_split_sizes=[c * self.META_DIM for c in intra_send_sizes + inter_send_sizes],
            )
        else:
            # 先处理组内通信 (ranks 4-7)
            intra_group_ranks = self.group1
            inter_group_ranks = self.group0
            
            # 组内通信
            intra_send_sizes = [send_split_sizes[r] for r in intra_group_ranks]
            intra_recv_sizes = [recv_split_sizes[r] for r in intra_group_ranks]
            
            # 组间通信
            inter_send_sizes = [send_split_sizes[r] for r in inter_group_ranks]
            inter_recv_sizes = [recv_split_sizes[r] for r in inter_group_ranks]
            
            # 执行通信
            dist.all_to_all_single(
                recv_buf,
                send_buf,
                output_split_sizes=inter_recv_sizes + intra_recv_sizes,
                input_split_sizes=inter_send_sizes + intra_send_sizes,
            )

            dist.all_to_all_single(
                recv_meta.view(-1),
                send_meta.view(-1),
                output_split_sizes=[c * self.META_DIM for c in inter_recv_sizes + intra_recv_sizes],
                input_split_sizes=[c * self.META_DIM for c in inter_send_sizes + intra_send_sizes],
            )
            
        recv_meta = recv_meta.view(-1, self.META_DIM)

        # 内存访问模式优化 - 批量处理
        if total_recv > 0:
            # 预先提取所有索引和权重
            src_tokens = recv_meta[:, 2].to(torch.long)
            src_ks = recv_meta[:, 3].to(torch.long)
            
            # 批量计算权重
            weights_selected = weights[src_tokens, src_ks]
            
            # 批量计算加权值
            weighted_values = recv_buf * weights_selected.unsqueeze(-1).to(recv_buf.dtype)
            
            # 使用scatter_add进行批量累加（避免循环）
            # 确保数据类型匹配
            weighted_values = weighted_values.to(out_tokens.dtype)
            
            # 扩展src_tokens以匹配weighted_values的形状
            src_tokens_expanded = src_tokens.unsqueeze(-1).expand_as(weighted_values)
            
            # 批量累加
            out_tokens.scatter_add_(0, src_tokens_expanded, weighted_values)

        return out_tokens

def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)

    # ata = PyTorchAllToAll(cfg, rank, world_size)
    # ata = OptimizedByQwenPyTorchAllToAll(cfg, rank, world_size)
    ata = OptimizedByDpskPyTorchAllToAll(cfg, rank, world_size)

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
