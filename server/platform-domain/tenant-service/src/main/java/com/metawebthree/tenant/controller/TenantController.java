package com.metawebthree.tenant.controller;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.tenant.dto.RegisterRequest;
import com.metawebthree.tenant.entity.Tenant;
import com.metawebthree.tenant.entity.TenantShop;
import com.metawebthree.tenant.entity.TenantUser;
import com.metawebthree.tenant.enums.TenantUserRole;
import com.metawebthree.tenant.service.CaptchaService;
import com.metawebthree.tenant.service.EmailVerificationService;
import com.metawebthree.tenant.service.TenantService;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;

import io.github.resilience4j.ratelimiter.annotation.RateLimiter;

import jakarta.validation.Valid;

import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/tenant")
public class TenantController {

    private final TenantService tenantService;
    private final CaptchaService captchaService;
    private final EmailVerificationService emailVerificationService;

    public TenantController(TenantService tenantService, CaptchaService captchaService,
                            EmailVerificationService emailVerificationService) {
        this.tenantService = tenantService;
        this.captchaService = captchaService;
        this.emailVerificationService = emailVerificationService;
    }

    @GetMapping("/captcha/generate")
    public ApiResponse<CaptchaService.CaptchaResult> generateCaptcha() {
        return ApiResponse.success(captchaService.generate());
    }

    @PostMapping("/email/send-verification-code")
    public ApiResponse<Void> sendVerificationCode(@RequestParam String email) {
        if (email == null || email.isBlank()) {
            return ApiResponse.error(ResponseStatus.PARAM_MISSING_ERROR, "Email is required");
        }
        if (tenantService.getByEmail(email) != null) {
            return ApiResponse.error(ResponseStatus.TENANT_ALREADY_EXISTS,
                "A tenant with this email already exists");
        }
        boolean sent = emailVerificationService.sendCode(email);
        if (!sent) {
            return ApiResponse.error(ResponseStatus.EMAIL_VERIFICATION_CODE_SEND_FAILED);
        }
        return ApiResponse.success();
    }

    @RateLimiter(name = "tenantRegister")
    @PostMapping("/register")
    public ApiResponse<Tenant> register(@Valid @RequestBody RegisterRequest request) {
        if (!captchaService.verify(request.getCaptchaToken(), request.getCaptchaAnswer())) {
            return ApiResponse.error(ResponseStatus.CAPTCHA_INVALID);
        }
        if (!emailVerificationService.verifyCode(request.getContactEmail(), request.getEmailCode())) {
            return ApiResponse.error(ResponseStatus.EMAIL_VERIFICATION_CODE_INVALID);
        }
        if (tenantService.getByCode(request.getCode()) != null) {
            return ApiResponse.error(ResponseStatus.TENANT_ALREADY_EXISTS,
                "A tenant with this code already exists");
        }
        if (tenantService.getByEmail(request.getContactEmail()) != null) {
            return ApiResponse.error(ResponseStatus.TENANT_ALREADY_EXISTS,
                "A tenant with this email already exists");
        }
        Tenant tenant = Tenant.builder()
            .name(request.getName())
            .code(request.getCode())
            .contactName(request.getContactName())
            .contactEmail(request.getContactEmail())
            .contactPhone(request.getContactPhone())
            .build();
        return ApiResponse.success(tenantService.create(tenant));
    }

    @PostMapping
    public ApiResponse<Tenant> create(@RequestBody Tenant tenant) {
        return ApiResponse.success(tenantService.create(tenant));
    }

    @PutMapping("/{id}")
    public ApiResponse<Tenant> update(@PathVariable Long id, @RequestBody Tenant tenant) {
        tenant.setId(id);
        return ApiResponse.success(tenantService.update(tenant));
    }

    @GetMapping("/{id}")
    public ApiResponse<Tenant> getById(@PathVariable Long id) {
        return ApiResponse.success(tenantService.getById(id));
    }

    @GetMapping("/code/{code}")
    public ApiResponse<Tenant> getByCode(@PathVariable String code) {
        return ApiResponse.success(tenantService.getByCode(code));
    }

    @GetMapping
    public ApiResponse<IPage<Tenant>> page(@RequestParam(defaultValue = "1") int page,
                                           @RequestParam(defaultValue = "10") int size,
                                           Tenant query) {
        return ApiResponse.success(tenantService.page(new Page<>(page, size), query));
    }

    @PostMapping("/{id}/approve")
    public ApiResponse<Void> approve(@PathVariable Long id) {
        tenantService.approve(id);
        return ApiResponse.success(null);
    }

    @PostMapping("/{id}/reject")
    public ApiResponse<Void> reject(@PathVariable Long id) {
        tenantService.reject(id);
        return ApiResponse.success(null);
    }

    @PostMapping("/{id}/disable")
    public ApiResponse<Void> disable(@PathVariable Long id) {
        tenantService.disable(id);
        return ApiResponse.success(null);
    }

    @GetMapping("/{id}/shop")
    public ApiResponse<TenantShop> getShop(@PathVariable Long id) {
        return ApiResponse.success(tenantService.getShopByTenant(id));
    }

    @PutMapping("/{id}/shop")
    public ApiResponse<TenantShop> updateShop(@PathVariable Long id, @RequestBody TenantShop shop) {
        shop.setTenantId(id);
        return ApiResponse.success(tenantService.updateShop(shop));
    }

    @PostMapping("/{id}/users")
    public ApiResponse<Void> associateUser(@PathVariable Long id,
                                           @RequestParam Long userId,
                                           @RequestParam TenantUserRole role) {
        tenantService.associateUser(id, userId, role);
        return ApiResponse.success(null);
    }

    @DeleteMapping("/{id}/users/{userId}")
    public ApiResponse<Void> removeUser(@PathVariable Long id, @PathVariable Long userId) {
        tenantService.removeUser(id, userId);
        return ApiResponse.success(null);
    }

    @GetMapping("/{id}/users")
    public ApiResponse<List<TenantUser>> getUsers(@PathVariable Long id) {
        return ApiResponse.success(tenantService.getUsersByTenant(id));
    }
}
