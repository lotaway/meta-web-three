package com.metawebthree.finance.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.common.ERPPermissions;
import com.metawebthree.finance.application.command.AccountSubjectCommandService;
import com.metawebthree.finance.application.query.AccountSubjectQueryService;
import com.metawebthree.finance.domain.entity.AccountSubject;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/finance/subjects")
public class AccountSubjectController {
    private final AccountSubjectCommandService commandService;
    private final AccountSubjectQueryService queryService;

    public AccountSubjectController(AccountSubjectCommandService commandService, 
            AccountSubjectQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    @RequirePermission(ERPPermissions.ACCOUNT_SUBJECT_CREATE)
    @PostMapping
    public ResponseEntity<Long> createSubject(@RequestBody SubjectCreateRequest request,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        Long id = commandService.createSubject(request.getSubjectCode(), request.getSubjectName(), 
            request.getDirection(), request.getParentId());
        return ResponseEntity.ok(id);
    }

    @RequirePermission(ERPPermissions.ACCOUNT_SUBJECT_READ)
    @GetMapping("/{id}")
    public ResponseEntity<AccountSubject> getSubject(@PathVariable Long id) {
        return queryService.getById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @RequirePermission(ERPPermissions.ACCOUNT_SUBJECT_READ)
    @GetMapping
    public ResponseEntity<List<AccountSubject>> listSubjects(@RequestParam(required = false) String status,
                                                              @RequestParam(required = false) Integer level) {
        List<AccountSubject> subjects;
        if ("ACTIVE".equalsIgnoreCase(status)) {
            subjects = queryService.listActiveSubjects();
        } else if (level != null) {
            subjects = queryService.listByLevel(level);
        } else {
            subjects = queryService.listAll();
        }
        return ResponseEntity.ok(subjects);
    }

    @RequirePermission(ERPPermissions.ACCOUNT_SUBJECT_READ)
    @GetMapping("/code/{code}")
    public ResponseEntity<AccountSubject> getByCode(@PathVariable String code) {
        return queryService.getBySubjectCode(code)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @RequirePermission(ERPPermissions.ACCOUNT_SUBJECT_READ)
    @GetMapping("/parent/{parentId}")
    public ResponseEntity<List<AccountSubject>> getByParentId(@PathVariable Long parentId) {
        return ResponseEntity.ok(queryService.listByParentId(parentId));
    }

    @RequirePermission(ERPPermissions.ACCOUNT_SUBJECT_UPDATE)
    @PostMapping("/{id}/disable")
    public ResponseEntity<Void> disable(@PathVariable Long id,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        commandService.disable(id);
        return ResponseEntity.ok().build();
    }

    @RequirePermission(ERPPermissions.ACCOUNT_SUBJECT_UPDATE)
    @PostMapping("/{id}/enable")
    public ResponseEntity<Void> enable(@PathVariable Long id,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        commandService.enable(id);
        return ResponseEntity.ok().build();
    }

    public static class SubjectCreateRequest {
        private String subjectCode;
        private String subjectName;
        private String direction;
        private Long parentId;

        public String getSubjectCode() { return subjectCode; }
        public void setSubjectCode(String subjectCode) { this.subjectCode = subjectCode; }
        public String getSubjectName() { return subjectName; }
        public void setSubjectName(String subjectName) { this.subjectName = subjectName; }
        public String getDirection() { return direction; }
        public void setDirection(String direction) { this.direction = direction; }
        public Long getParentId() { return parentId; }
        public void setParentId(Long parentId) { this.parentId = parentId; }
    }
}