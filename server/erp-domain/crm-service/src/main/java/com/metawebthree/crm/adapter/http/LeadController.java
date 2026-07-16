package com.metawebthree.crm.adapter.http;

import com.metawebthree.crm.adapter.client.UserServiceClient;
import com.metawebthree.crm.adapter.vo.Result;
import com.metawebthree.crm.application.command.LeadCommandService;
import com.metawebthree.crm.application.query.LeadQueryService;
import com.metawebthree.crm.domain.entity.Lead;
import com.metawebthree.crm.domain.entity.Opportunity;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/crm/leads")
@RequiredArgsConstructor
public class LeadController {

    private final LeadQueryService leadQueryService;
    private final LeadCommandService leadCommandService;
    private final UserServiceClient userServiceClient;

    @GetMapping("/{id}")
    public Result<Lead> getById(@PathVariable Long id) {
        return Result.success(leadQueryService.getById(id));
    }

    @GetMapping("/list")
    public Result<List<Lead>> listAll() {
        return Result.success(leadQueryService.listAll());
    }

    @GetMapping("/status/{status}")
    public Result<List<Lead>> listByStatus(@PathVariable String status) {
        return Result.success(leadQueryService.listByStatus(status));
    }

    @GetMapping("/source/{source}")
    public Result<List<Lead>> listBySource(@PathVariable String source) {
        return Result.success(leadQueryService.listBySource(source));
    }

    @GetMapping("/assigned/{assignedTo}")
    public Result<List<Lead>> listByAssignedTo(@PathVariable String assignedTo) {
        return Result.success(leadQueryService.listByAssignedTo(assignedTo));
    }

    @GetMapping("/search")
    public Result<List<Lead>> search(@RequestParam String keywords) {
        return Result.success(leadQueryService.search(keywords));
    }

    @PostMapping
    public Result<Lead> create(@RequestBody Lead lead) {
        return Result.success(leadCommandService.create(lead));
    }

    @PutMapping
    public Result<Lead> update(@RequestBody Lead lead) {
        return Result.success(leadCommandService.update(lead));
    }

    @DeleteMapping("/{id}")
    public Result<Void> delete(@PathVariable Long id) {
        leadCommandService.delete(id);
        return Result.success();
    }

    @PostMapping("/{id}/convert")
    public Result<Opportunity> convert(@PathVariable Long id, @RequestBody Opportunity opportunity) {
        return Result.success(leadCommandService.convert(id, opportunity));
    }

    @PostMapping("/{id}/disqualify")
    public Result<Void> disqualify(@PathVariable Long id, @RequestParam String reason) {
        leadCommandService.disqualify(id, reason);
        return Result.success();
    }

    @GetMapping("/sync/users")
    public Result<List<Map<String, Object>>> syncUsers(@RequestParam(required = false) String keyword) {
        List<Map<String, Object>> users = keyword != null ?
                userServiceClient.searchUsers(keyword) :
                userServiceClient.searchUsers("");
        return Result.success(users);
    }

    @GetMapping("/sync/user/{userId}")
    public Result<Map<String, Object>> syncUserById(@PathVariable Long userId) {
        Map<String, Object> user = userServiceClient.getUserById(userId);
        return user.isEmpty() ? Result.error("User not found") : Result.success(user);
    }
}
