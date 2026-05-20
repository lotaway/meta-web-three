package com.metawebthree.user.application;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.common.generated.rpc.UserRiskProfile;
import com.metawebthree.common.generated.rpc.UserRiskProfileService;
import com.metawebthree.common.generated.rpc.GetUserRiskProfileRequest;
import com.metawebthree.common.generated.rpc.GetUserRiskProfileResponse;
import com.metawebthree.common.generated.rpc.UpdateRiskProfileRequest;
import com.metawebthree.common.generated.rpc.UpdateRiskProfileResponse;
import com.metawebthree.common.generated.rpc.DeviceRiskTag;
import com.metawebthree.user.domain.model.UserProfileDO;
import com.metawebthree.user.infrastructure.persistence.mapper.UserProfileMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.util.concurrent.CompletableFuture;

import org.apache.dubbo.config.annotation.DubboService;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Slf4j
@Service
@DubboService
@RequiredArgsConstructor
public class UserRiskProfileServiceImpl implements UserRiskProfileService {

    private final UserProfileMapper userProfileMapper;

    @Override
    public GetUserRiskProfileResponse getUserRiskProfile(GetUserRiskProfileRequest request) {
        Long userId = request.getUserId();
        log.info("Fetching risk profile for user: {}", userId);

        UserProfileDO profileDO = userProfileMapper.selectOne(
                new LambdaQueryWrapper<UserProfileDO>().eq(UserProfileDO::getUserId, userId));

        UserRiskProfile profile;
        if (profileDO == null) {
            log.warn("User risk profile not found for user: {}, returning default", userId);
            profile = UserRiskProfile.newBuilder()
                    .setUserId(userId)
                    .setAge(0)
                    .setExternalDebtRatio(0.0f)
                    .setGpsStability(0.0f)
                    .setDeviceSharedDegree(0)
                    .setDeviceRiskTag(DeviceRiskTag.UNKNOWN)
                    .build();
        } else {
            profile = UserRiskProfile.newBuilder()
                    .setUserId(profileDO.getUserId())
                    .setAge(profileDO.getAge() != null ? profileDO.getAge() : 0)
                    .setExternalDebtRatio(
                            profileDO.getExternalDebtRatio() != null ? profileDO.getExternalDebtRatio() : 0.0f)
                    .setGpsStability(profileDO.getGpsStability() != null ? profileDO.getGpsStability() : 0.0f)
                    .setDeviceSharedDegree(
                            profileDO.getDeviceSharedDegree() != null ? profileDO.getDeviceSharedDegree() : 0)
                    .setDeviceRiskTag(parseDeviceRiskTag(profileDO.getDeviceRiskTag()))
                    .build();
        }

        return GetUserRiskProfileResponse.newBuilder().setProfile(profile).build();
    }

    @Override
    @Transactional
    public UpdateRiskProfileResponse updateRiskProfile(UpdateRiskProfileRequest request) {
        Long userId = request.getUserId();
        if (userId == null) {
            log.error("Cannot update risk profile: userId is null");
            return UpdateRiskProfileResponse.newBuilder().setSuccess(false).build();
        }

        log.info("Updating risk profile for user: {}", userId);

        UserProfileDO existingProfile = userProfileMapper.selectOne(
                new LambdaQueryWrapper<UserProfileDO>().eq(UserProfileDO::getUserId, userId));

        if (existingProfile == null) {
            UserProfileDO newProfile = UserProfileDO.builder()
                    .userId(userId)
                    .age(request.getAge())
                    .externalDebtRatio(request.getExternalDebtRatio())
                    .gpsStability(request.getGpsStability())
                    .deviceSharedDegree(request.getDeviceSharedDegree())
                    .deviceRiskTag(request.getDeviceRiskTag().name())
                    .build();
            userProfileMapper.insert(newProfile);
        } else {
            if (request.getAge() != 0)
                existingProfile.setAge(request.getAge());
            if (request.getExternalDebtRatio() != 0.0f)
                existingProfile.setExternalDebtRatio(request.getExternalDebtRatio());
            if (request.getGpsStability() != 0.0f)
                existingProfile.setGpsStability(request.getGpsStability());
            if (request.getDeviceSharedDegree() != 0)
                existingProfile.setDeviceSharedDegree(request.getDeviceSharedDegree());
            if (request.getDeviceRiskTag() != DeviceRiskTag.UNKNOWN)
                existingProfile.setDeviceRiskTag(request.getDeviceRiskTag().name());

            userProfileMapper.updateById(existingProfile);
        }

        return UpdateRiskProfileResponse.newBuilder().setSuccess(true).build();
    }

    private DeviceRiskTag parseDeviceRiskTag(String tag) {
        if (tag == null || tag.isEmpty()) {
            return DeviceRiskTag.UNKNOWN;
        }
        try {
            return DeviceRiskTag.valueOf(tag);
        } catch (IllegalArgumentException e) {
            return DeviceRiskTag.UNKNOWN;
        }
    }

    @Override
    public CompletableFuture<GetUserRiskProfileResponse> getUserRiskProfileAsync(GetUserRiskProfileRequest request) {
        return CompletableFuture.supplyAsync(() -> getUserRiskProfile(request));
    }

    @Override
    public CompletableFuture<UpdateRiskProfileResponse> updateRiskProfileAsync(UpdateRiskProfileRequest request) {
        return CompletableFuture.supplyAsync(() -> updateRiskProfile(request));
    }
}
