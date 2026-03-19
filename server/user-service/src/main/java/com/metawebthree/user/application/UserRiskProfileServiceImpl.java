package com.metawebthree.user.application;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.common.dto.UserRiskProfileDTO;
import com.metawebthree.common.dto.UserRiskProfileUpdateDTO;
import com.metawebthree.common.rpc.UserRiskProfileService;
import com.metawebthree.user.domain.model.UserProfileDO;
import com.metawebthree.user.infrastructure.persistence.mapper.UserProfileMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
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
    public UserRiskProfileDTO getUserRiskProfile(Long userId) {
        log.info("Fetching risk profile for user: {}", userId);
        UserProfileDO profileDO = userProfileMapper.selectOne(
                new LambdaQueryWrapper<UserProfileDO>().eq(UserProfileDO::getUserId, userId)
        );

        if (profileDO == null) {
            log.warn("User risk profile not found for user: {}, returning default", userId);
            // Return default/empty profile
            return UserRiskProfileDTO.builder()
                    .userId(userId)
                    .age(0) // Default age
                    .externalDebtRatio(0.0f)
                    .gpsStability(0.0f)
                    .deviceSharedDegree(0)
                    .deviceRiskTag("UNKNOWN")
                    .build();
        }

        return UserRiskProfileDTO.builder()
                .userId(profileDO.getUserId())
                .age(profileDO.getAge())
                .externalDebtRatio(profileDO.getExternalDebtRatio())
                .gpsStability(profileDO.getGpsStability())
                .deviceSharedDegree(profileDO.getDeviceSharedDegree())
                .deviceRiskTag(profileDO.getDeviceRiskTag())
                .build();
    }

    @Override
    @Transactional
    public void updateRiskProfile(UserRiskProfileUpdateDTO updateDTO) {
        Long userId = updateDTO.getUserId();
        if (userId == null) {
            log.error("Cannot update risk profile: userId is null");
            return;
        }

        log.info("Updating risk profile for user: {}", userId);
        
        UserProfileDO existingProfile = userProfileMapper.selectOne(
                new LambdaQueryWrapper<UserProfileDO>().eq(UserProfileDO::getUserId, userId)
        );

        if (existingProfile == null) {
            // Create new profile
            UserProfileDO newProfile = UserProfileDO.builder()
                    .userId(userId)
                    .age(updateDTO.getAge())
                    .externalDebtRatio(updateDTO.getExternalDebtRatio())
                    .gpsStability(updateDTO.getGpsStability())
                    .deviceSharedDegree(updateDTO.getDeviceSharedDegree())
                    .deviceRiskTag(updateDTO.getDeviceRiskTag())
                    .build();
            userProfileMapper.insert(newProfile);
        } else {
            // Update existing profile (only non-null fields)
            if (updateDTO.getAge() != null) existingProfile.setAge(updateDTO.getAge());
            if (updateDTO.getExternalDebtRatio() != null) existingProfile.setExternalDebtRatio(updateDTO.getExternalDebtRatio());
            if (updateDTO.getGpsStability() != null) existingProfile.setGpsStability(updateDTO.getGpsStability());
            if (updateDTO.getDeviceSharedDegree() != null) existingProfile.setDeviceSharedDegree(updateDTO.getDeviceSharedDegree());
            if (updateDTO.getDeviceRiskTag() != null) existingProfile.setDeviceRiskTag(updateDTO.getDeviceRiskTag());
            
            userProfileMapper.updateById(existingProfile);
        }
    }
}
