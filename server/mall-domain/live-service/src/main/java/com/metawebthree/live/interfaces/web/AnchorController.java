package com.metawebthree.live.interfaces.web;

import com.metawebthree.live.application.LiveApplicationService;
import com.metawebthree.live.domain.model.*;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/live/anchor")
public class AnchorController {

    private final LiveApplicationService liveApplicationService;

    public AnchorController(LiveApplicationService liveApplicationService) {
        this.liveApplicationService = liveApplicationService;
    }

    @PostMapping("/create")
    public Anchor createAnchor(@RequestBody Map<String, String> request) {
        return liveApplicationService.createAnchor(
                Long.parseLong(request.get("userId")),
                request.get("anchorName"),
                request.get("avatar"),
                request.get("description")
        );
    }

    @GetMapping("/{id}")
    public Anchor getAnchorById(@PathVariable Long id) {
        return liveApplicationService.getAnchorById(id);
    }

    @GetMapping("/user/{userId}")
    public Anchor getAnchorByUserId(@PathVariable Long userId) {
        return liveApplicationService.getAnchorByUserId(userId);
    }
}