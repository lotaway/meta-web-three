package com.metawebthree.tenant.controller;

import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.tenant.dto.RegisterRequest;
import com.metawebthree.tenant.entity.Tenant;
import com.metawebthree.tenant.entity.TenantShop;
import com.metawebthree.tenant.entity.TenantUser;
import com.metawebthree.tenant.enums.TenantUserRole;
import com.metawebthree.tenant.service.CaptchaService;
import com.metawebthree.tenant.service.EmailVerificationService;
import com.metawebthree.tenant.service.IpRateLimitService;
import com.metawebthree.tenant.service.TenantService;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;

import io.github.resilience4j.ratelimiter.annotation.RateLimiter;

import jakarta.validation.Valid;
import jakarta.servlet.http.HttpServletRequest;

import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
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
    private final IpRateLimitService ipRateLimitService;

    public TenantController(TenantService tenantService, CaptchaService captchaService,
                            EmailVerificationService emailVerificationService,
                            IpRateLimitService ipRateLimitService) {
        this.tenantService = tenantService;
        this.captchaService = captchaService;
        this.emailVerificationService = emailVerificationService;
        this.ipRateLimitService = ipRateLimitService;
    }

    @GetMapping("/captcha/generate")
    public ApiResponse<CaptchaService.CaptchaResult> generateCaptcha(HttpServletRequest request) {
        if (!ipRateLimitService.isAllowed(request)) {
            return ApiResponse.error(ResponseStatus.FORBIDDEN, "Too many requests");
        }
        return ApiResponse.success(captchaService.generate());
    }

    @PostMapping("/email/send-verification-code")
    public ApiResponse<Void> sendVerificationCode(HttpServletRequest request, @RequestParam String email) {
        if (!ipRateLimitService.isAllowed(request)) {
            return ApiResponse.error(ResponseStatus.FORBIDDEN, "Too many requests");
        }
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
    public ApiResponse<Tenant> register(HttpServletRequest request, @Valid @RequestBody RegisterRequest registerRequest) {
        if (!ipRateLimitService.isAllowed(request)) {
            return ApiResponse.error(ResponseStatus.FORBIDDEN, "Too many requests");
        }
        if (!captchaService.verify(registerRequest.getCaptchaToken(), registerRequest.getCaptchaAnswer())) {
            return ApiResponse.error(ResponseStatus.CAPTCHA_INVALID);
        }
        if (!emailVerificationService.verifyCode(registerRequest.getContactEmail(), registerRequest.getEmailCode())) {
            return ApiResponse.error(ResponseStatus.EMAIL_VERIFICATION_CODE_INVALID);
        }
        if (tenantService.getByCode(registerRequest.getCode()) != null) {
            return ApiResponse.error(ResponseStatus.TENANT_ALREADY_EXISTS,
                "A tenant with this code already exists");
        }
        if (tenantService.getByEmail(registerRequest.getContactEmail()) != null) {
            return ApiResponse.error(ResponseStatus.TENANT_ALREADY_EXISTS,
                "A tenant with this email already exists");
        }
        Tenant tenant = Tenant.builder()
            .name(registerRequest.getName())
            .code(registerRequest.getCode())
            .contactName(registerRequest.getContactName())
            .contactEmail(registerRequest.getContactEmail())
            .contactPhone(registerRequest.getContactPhone())
            .build();
        return ApiResponse.success(tenantService.create(tenant));
    }

    @PostMapping
    public ApiResponse<Tenant> create(@RequestHeader(HeaderConstants.USER_ROLE) String userRole,
                                      @RequestBody Tenant tenant) {
        if (!isAdmin(userRole)) {
            return ApiResponse.error(ResponseStatus.FORBIDDEN);
        }
        return ApiResponse.success(tenantService.create(tenant));
    }

    @PutMapping("/{id}")
    public ApiResponse<Tenant> update(@RequestHeader(HeaderConstants.USER_ROLE) String userRole,
                                      @PathVariable Long id, @RequestBody Tenant tenant) {
        if (!isAdmin(userRole)) {
            return ApiResponse.error(ResponseStatus.FORBIDDEN);
        }
        tenant.setId(id);
        return ApiResponse.success(tenantService.update(tenant));
    }

    @GetMapping("/{id}")
    public ApiResponse<Tenant> getById(@RequestHeader(HeaderConstants.USER_ROLE) String userRole,
                                       @PathVariable Long id) {
        if (!isAdmin(userRole)) {
            return ApiResponse.error(ResponseStatus.FORBIDDEN);
        }
        return ApiResponse.success(tenantService.getById(id));
    }

    @GetMapping("/code/{code}")
    public ApiResponse<Tenant> getByCode(@RequestHeader(HeaderConstants.USER_ROLE) String userRole,
                                         @PathVariable String code) {
        if (!isAdmin(userRole)) {
            return ApiResponse.error(ResponseStatus.FORBIDDEN);
        }
        return ApiResponse.success(tenantService.getByCode(code));
    }

    @GetMapping
    public ApiResponse<IPage<Tenant>> page(@RequestHeader(HeaderConstants.USER_ROLE) String userRole,
                                           @RequestParam(defaultValue = "1") int page,
                                           @RequestParam(defaultValue = "10") int size,
                                           Tenant query) {
        if (!isAdmin(userRole)) {
            return ApiResponse.error(ResponseStatus.FORBIDDEN);
        }
        return ApiResponse.success(tenantService.page(new Page<>(page, size), query));
    }

    @PostMapping("/{id}/approve")
    public ApiResponse<Void> approve(@RequestHeader(HeaderConstants.USER_ROLE) String userRole,
                                     @PathVariable Long id) {
        if (!isAdmin(userRole)) {
            return ApiResponse.error(ResponseStatus.FORBIDDEN);
        }
        tenantService.approve(id);
        return ApiResponse.success(null);
    }

    @PostMapping("/{id}/reject")
    public ApiResponse<Void> reject(@RequestHeader(HeaderConstants.USER_ROLE) String userRole,
                                    @PathVariable Long id) {
        if (!isAdmin(userRole)) {
            return ApiResponse.error(ResponseStatus.FORBIDDEN);
        }
        tenantService.reject(id);
        return ApiResponse.success(null);
    }

    @PostMapping("/{id}/disable")
    public ApiResponse<Void> disable(@RequestHeader(HeaderConstants.USER_ROLE) String userRole,
                                     @PathVariable Long id) {
        if (!isAdmin(userRole)) {
            return ApiResponse.error(ResponseStatus.FORBIDDEN);
        }
        tenantService.disable(id);
        return ApiResponse.success(null);
    }

    @GetMapping("/{id}/shop")
    public ApiResponse<TenantShop> getShop(@RequestHeader(HeaderConstants.USER_ROLE) String userRole,
                                           @PathVariable Long id) {
        if (!isMerchantOrAdmin(userRole)) {
            return ApiResponse.error(ResponseStatus.FORBIDDEN);
        }
        return ApiResponse.success(tenantService.getShopByTenant(id));
    }

    @PutMapping("/{id}/shop")
    public ApiResponse<TenantShop> updateShop(@RequestHeader(HeaderConstants.USER_ROLE) String userRole,
                                              @PathVariable Long id, @RequestBody TenantShop shop) {
        if (!isMerchantOrAdmin(userRole)) {
            return ApiResponse.error(ResponseStatus.FORBIDDEN);
        }
        shop.setTenantId(id);
        return ApiResponse.success(tenantService.updateShop(shop));
    }

    @PostMapping("/{id}/users")
    public ApiResponse<Void> associateUser(@RequestHeader(HeaderConstants.USER_ROLE) String userRole,
                                           @PathVariable Long id,
                                           @RequestParam Long userId,
                                           @RequestParam TenantUserRole role) {
        if (!isMerchantOrAdmin(userRole)) {
            return ApiResponse.error(ResponseStatus.FORBIDDEN);
        }
        tenantService.associateUser(id, userId, role);
        return ApiResponse.success(null);
    }

    @DeleteMapping("/{id}/users/{userId}")
    public ApiResponse<Void> removeUser(@RequestHeader(HeaderConstants.USER_ROLE) String userRole,
                                        @PathVariable Long id, @PathVariable Long userId) {
        if (!isAdmin(userRole)) {
            return ApiResponse.error(ResponseStatus.FORBIDDEN);
        }
        tenantService.removeUser(id, userId);
        return ApiResponse.success(null);
    }

    @GetMapping("/{id}/users")
    public ApiResponse<List<TenantUser>> getUsers(@RequestHeader(HeaderConstants.USER_ROLE) String userRole,
                                                  @PathVariable Long id) {
        if (!isMerchantOrAdmin(userRole)) {
            return ApiResponse.error(ResponseStatus.FORBIDDEN);
        }
        return ApiResponse.success(tenantService.getUsersByTenant(id));
    }

    private boolean isAdmin(String userRole) {
        return "ADMIN".equals(userRole);
    }

    private boolean isMerchantOrAdmin(String userRole) {
        return "ADMIN".equals(userRole) || "MERCHANT".equals(userRole);
    }
}
