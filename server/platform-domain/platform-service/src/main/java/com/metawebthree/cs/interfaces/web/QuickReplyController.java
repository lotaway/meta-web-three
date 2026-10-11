package com.metawebthree.cs.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.cs.application.QuickReplyService;
import com.metawebthree.cs.domain.model.QuickReply;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/cs/quick-reply")
@Tag(name = "Quick Reply Controller", description = "快捷回复接口")
public class QuickReplyController {
    private final QuickReplyService quickReplyService;

    public QuickReplyController(QuickReplyService quickReplyService) {
        this.quickReplyService = quickReplyService;
    }

    @Operation(summary = "创建快捷回复")
    @PostMapping("/create")
    public ApiResponse<QuickReply> create(@RequestBody QuickReply request) {
        return ApiResponse.success(
                quickReplyService.create(request.getGroupId(), request.getTitle(),
                        request.getContent(), request.getMsgType()));
    }

    @Operation(summary = "删除快捷回复")
    @DeleteMapping("/delete")
    public ApiResponse<Void> delete(@RequestParam Long id) {
        quickReplyService.delete(id);
        return ApiResponse.success();
    }

    @Operation(summary = "根据分组查询")
    @GetMapping("/list")
    public ApiResponse<List<QuickReply>> list(@RequestParam(required = false) Long groupId) {
        if (groupId != null) {
            return ApiResponse.success(quickReplyService.listByGroup(groupId));
        }
        return ApiResponse.success(quickReplyService.listAll());
    }
}
