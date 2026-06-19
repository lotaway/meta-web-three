package com.metawebthree.dataanalysis.application.query;

import com.metawebthree.dataanalysis.application.dto.*;
import com.metawebthree.dataanalysis.domain.entity.UserProfileDO;
import com.metawebthree.dataanalysis.infrastructure.persistence.mapper.UserProfileMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class UserPortraitQueryService {

    private final UserProfileMapper userProfileMapper;

    public UserPortraitDTO getUserPortrait(String startDate, String endDate) {
        List<UserProfileDO> profiles = userProfileMapper.selectAll();
        
        UserPortraitDTO result = new UserPortraitDTO();
        result.setTotalUsers((long) profiles.size());
        result.setNewUsers(userProfileMapper.countNewUsers(startDate, endDate));
        result.setActiveUsers(userProfileMapper.countActiveUsers(startDate, endDate));
        result.setAvgAge(calculateAvgAge(profiles));
        result.setGenderRatio(calculateGenderRatio(profiles));
        result.setRegionDistribution(getRegionDistribution(profiles));
        result.setCategoryPreference(getCategoryPreference(profiles));
        
        return result;
    }

    public UserProfileDTO getUserProfile(Long userId) {
        UserProfileDO profile = userProfileMapper.selectByUserId(userId);
        return profile != null ? toDTO(profile) : null;
    }

    private int calculateAvgAge(List<UserProfileDO> profiles) {
        if (profiles.isEmpty()) return 0;
        int sum = profiles.stream()
            .filter(p -> p.getAge() != null)
            .mapToInt(UserProfileDO::getAge)
            .sum();
        return profiles.size() > 0 ? sum / profiles.size() : 0;
    }

    private String calculateGenderRatio(List<UserProfileDO> profiles) {
        long male = profiles.stream().filter(p -> "M".equals(p.getGender())).count();
        long female = profiles.stream().filter(p -> "F".equals(p.getGender())).count();
        long total = profiles.size();
        if (total == 0) return "0:0";
        return male + ":" + female;
    }

    private List<RegionDistributionDTO> getRegionDistribution(List<UserProfileDO> profiles) {
        Map<String, Long> regionMap = profiles.stream()
            .collect(Collectors.groupingBy(UserProfileDO::getRegion, Collectors.counting()));
        
        List<RegionDistributionDTO> result = new ArrayList<>();
        for (Map.Entry<String, Long> entry : regionMap.entrySet()) {
            RegionDistributionDTO dto = new RegionDistributionDTO();
            dto.setRegion(entry.getKey());
            dto.setUserCount(entry.getValue());
            dto.setProportion(entry.getValue() * 100.0 / profiles.size());
            result.add(dto);
        }
        return result;
    }

    private List<CategoryPreferenceDTO> getCategoryPreference(List<UserProfileDO> profiles) {
        Map<String, Long> categoryMap = profiles.stream()
            .filter(p -> p.getCategoryPreference() != null)
            .collect(Collectors.groupingBy(UserProfileDO::getCategoryPreference, Collectors.counting()));
        
        List<CategoryPreferenceDTO> result = new ArrayList<>();
        for (Map.Entry<String, Long> entry : categoryMap.entrySet()) {
            CategoryPreferenceDTO dto = new CategoryPreferenceDTO();
            dto.setCategory(entry.getKey());
            dto.setUserCount(entry.getValue());
            dto.setProportion(entry.getValue() * 100.0 / profiles.size());
            result.add(dto);
        }
        return result;
    }

    private UserProfileDTO toDTO(UserProfileDO profile) {
        UserProfileDTO dto = new UserProfileDTO();
        dto.setUserId(profile.getUserId());
        dto.setAge(profile.getAge());
        dto.setGender(profile.getGender());
        dto.setRegion(profile.getRegion());
        dto.setPreference(profile.getPreference());
        dto.setPurchaseFrequency(profile.getPurchaseFrequency());
        dto.setTotalSpent(profile.getTotalSpent());
        dto.setOrderCount(profile.getOrderCount());
        dto.setCategoryPreference(profile.getCategoryPreference());
        return dto;
    }
}