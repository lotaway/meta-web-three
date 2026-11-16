package com.metawebthree.common.autoconfigure;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnWebApplication;
import org.springframework.context.annotation.Import;
import org.springframework.data.redis.core.StringRedisTemplate;

import com.metawebthree.common.VO.PageConfigVO;
import com.metawebthree.common.auth.TokenBlacklistService;
import com.metawebthree.common.utils.JwtUtil;
import com.metawebthree.common.utils.UserJwtUtil;

/**
 * Shared authentication helpers used by servlet services. Security-related
 * beans are on by default; only storage-facing beans require an opt-in.
 */
@AutoConfiguration
@ConditionalOnWebApplication(type = ConditionalOnWebApplication.Type.SERVLET)
@ConditionalOnClass(StringRedisTemplate.class)
@Import({TokenBlacklistService.class, PageConfigVO.class, JwtUtil.class, UserJwtUtil.class})
public class CommonSecurityBeansAutoConfiguration {
}
