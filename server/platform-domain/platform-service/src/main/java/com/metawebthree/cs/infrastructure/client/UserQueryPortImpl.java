package com.metawebthree.cs.infrastructure.client;

import java.util.Optional;

import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import com.metawebthree.common.generated.rpc.GetUserPhoneRequest;
import com.metawebthree.common.generated.rpc.GetUserPhoneResponse;
import com.metawebthree.common.generated.rpc.UserService;
import com.metawebthree.cs.domain.ports.UserQueryPort;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;

@Component
@Slf4j
public class UserQueryPortImpl implements UserQueryPort {

    @DubboReference(check = false, lazy = true)
    private UserService userService;

    private final RestTemplate restTemplate;

    @Value("${user-service.url:http://user-service:10083}")
    private String userServiceUrl;

    public UserQueryPortImpl(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    @Override
    public Optional<String> findNickname(Long userId) {
        try {
            UserInfoDTO userInfo = getUserInfo(userId);
            if (userInfo != null && userInfo.getNickname() != null) {
                return Optional.of(userInfo.getNickname());
            }
        } catch (Exception e) {
            log.warn("Failed to fetch nickname for user {}: {}", userId, e.getMessage());
        }
        return Optional.empty();
    }

    @Override
    public Optional<String> findAvatar(Long userId) {
        try {
            UserInfoDTO userInfo = getUserInfo(userId);
            if (userInfo != null && userInfo.getAvatar() != null) {
                return Optional.of(userInfo.getAvatar());
            }
        } catch (Exception e) {
            log.warn("Failed to fetch avatar for user {}: {}", userId, e.getMessage());
        }
        return Optional.empty();
    }

    @Override
    public Optional<String> findPhone(Long userId) {
        // Use Dubbo RPC to get phone number
        try {
            GetUserPhoneRequest request = GetUserPhoneRequest.newBuilder()
                    .setUserId(userId)
                    .build();
            GetUserPhoneResponse response = userService.getUserPhone(request);
            if (response != null && response.getSuccess() && !response.getPhone().isEmpty()) {
                return Optional.of(response.getPhone());
            }
        } catch (Exception e) {
            log.warn("Failed to fetch phone for user {} via RPC: {}", userId, e.getMessage());
        }
        return Optional.empty();
    }

    private UserInfoDTO getUserInfo(Long userId) {
        try {
            String url = userServiceUrl + "/user/info-by-id?id=" + userId;
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            // For internal service calls, we pass userId as a header
            headers.set("X-User-Id", String.valueOf(userId));
            
            HttpEntity<String> entity = new HttpEntity<>(headers);
            var response = restTemplate.exchange(url, HttpMethod.GET, entity, ApiResponse.class);
            
            if (response.getBody() != null) {
                ApiResponse apiResponse = (ApiResponse) response.getBody();
                if (apiResponse.getCode() == 0 && apiResponse.getData() != null) {
                    return parseUserInfo(apiResponse.getData());
                }
            }
        } catch (Exception e) {
            log.debug("getUserInfo failed via /user/info-by-id, trying alternative method: {}", e.getMessage());
        }
        return null;
    }

    private UserInfoDTO parseUserInfo(Object data) {
        if (data == null) {
            return null;
        }
        UserInfoDTO dto = new UserInfoDTO();
        if (data instanceof java.util.Map) {
            java.util.Map map = (java.util.Map) data;
            Object nickname = map.get("nickname");
            if (nickname != null) {
                dto.setNickname(nickname.toString());
            }
            Object avatar = map.get("avatar");
            if (avatar != null) {
                dto.setAvatar(avatar.toString());
            }
        }
        return dto;
    }

    @Data
    private static class UserInfoDTO {
        private Long id;
        private String nickname;
        private String avatar;
        private String email;
    }

    @Data
    private static class ApiResponse {
        private int code;
        private String message;
        private Object data;
    }
}