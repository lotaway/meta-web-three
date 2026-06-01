package com.metawebthree.live.interfaces.web;

import com.metawebthree.live.application.LiveApplicationService;
import com.metawebthree.live.domain.model.LiveRoom;
import com.metawebthree.live.domain.model.LiveProduct;
import com.metawebthree.live.domain.model.LiveComment;
import com.metawebthree.live.domain.model.LiveOrder;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/live/room")
public class LiveRoomController {

    private final LiveApplicationService liveApplicationService;

    public LiveRoomController(LiveApplicationService liveApplicationService) {
        this.liveApplicationService = liveApplicationService;
    }

    @PostMapping("/start")
    public LiveRoom startLiveRoom(@RequestBody Map<String, String> request) {
        return liveApplicationService.startLiveRoom(
                Long.parseLong(request.get("anchorId")),
                request.get("roomName"),
                request.get("coverImage"),
                request.get("description")
        );
    }

    @PostMapping("/end/{roomId}")
    public LiveRoom endLiveRoom(@PathVariable Long roomId) {
        return liveApplicationService.endLiveRoom(roomId);
    }

    @GetMapping("/{id}")
    public LiveRoom getLiveRoomById(@PathVariable Long id) {
        return liveApplicationService.getLiveRoomById(id);
    }

    @GetMapping("/anchor/{anchorId}")
    public List<LiveRoom> getLiveRoomsByAnchorId(@PathVariable Long anchorId) {
        return liveApplicationService.getLiveRoomsByAnchorId(anchorId);
    }

    @PostMapping("/product/attach")
    public LiveProduct attachProduct(@RequestBody Map<String, Object> request) {
        return liveApplicationService.attachProduct(
                Long.parseLong((String) request.get("roomId")),
                Long.parseLong((String) request.get("productId")),
                Long.parseLong((String) request.get("price")),
                Long.parseLong((String) request.get("discountPrice")),
                Integer.parseInt((String) request.get("stock"))
        );
    }

    @GetMapping("/{roomId}/products")
    public List<LiveProduct> getLiveProductsByRoomId(@PathVariable Long roomId) {
        return liveApplicationService.getLiveProductsByRoomId(roomId);
    }

    @PostMapping("/comment")
    public LiveComment postComment(@RequestBody Map<String, String> request) {
        return liveApplicationService.postComment(
                Long.parseLong(request.get("roomId")),
                Long.parseLong(request.get("userId")),
                request.get("userName"),
                request.get("content")
        );
    }

    @GetMapping("/{roomId}/comments")
    public List<LiveComment> getLiveCommentsByRoomId(@PathVariable Long roomId) {
        return liveApplicationService.getLiveCommentsByRoomId(roomId);
    }

    @PostMapping("/order/create")
    public LiveOrder createLiveOrder(@RequestBody Map<String, String> request) {
        return liveApplicationService.createLiveOrder(
                Long.parseLong(request.get("roomId")),
                Long.parseLong(request.get("productId")),
                Long.parseLong(request.get("userId")),
                Integer.parseInt(request.get("quantity"))
        );
    }

    @GetMapping("/{roomId}/orders")
    public List<LiveOrder> getLiveOrdersByRoomId(@PathVariable Long roomId) {
        return liveApplicationService.getLiveOrdersByRoomId(roomId);
    }

    @GetMapping("/user/{userId}/orders")
    public List<LiveOrder> getLiveOrdersByUserId(@PathVariable Long userId) {
        return liveApplicationService.getLiveOrdersByUserId(userId);
    }
}