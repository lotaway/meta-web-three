package com.metawebthree.dataanalysis.application.dto;

import java.util.List;

public class UserPortraitDTO {
    private Long totalUsers;
    private Long newUsers;
    private Long activeUsers;
    private Integer avgAge;
    private String genderRatio;
    private List<RegionDistributionDTO> regionDistribution;
    private List<CategoryPreferenceDTO> categoryPreference;

    public Long getTotalUsers() {
        return totalUsers;
    }

    public void setTotalUsers(Long totalUsers) {
        this.totalUsers = totalUsers;
    }

    public Long getNewUsers() {
        return newUsers;
    }

    public void setNewUsers(Long newUsers) {
        this.newUsers = newUsers;
    }

    public Long getActiveUsers() {
        return activeUsers;
    }

    public void setActiveUsers(Long activeUsers) {
        this.activeUsers = activeUsers;
    }

    public Integer getAvgAge() {
        return avgAge;
    }

    public void setAvgAge(Integer avgAge) {
        this.avgAge = avgAge;
    }

    public String getGenderRatio() {
        return genderRatio;
    }

    public void setGenderRatio(String genderRatio) {
        this.genderRatio = genderRatio;
    }

    public List<RegionDistributionDTO> getRegionDistribution() {
        return regionDistribution;
    }

    public void setRegionDistribution(List<RegionDistributionDTO> regionDistribution) {
        this.regionDistribution = regionDistribution;
    }

    public List<CategoryPreferenceDTO> getCategoryPreference() {
        return categoryPreference;
    }

    public void setCategoryPreference(List<CategoryPreferenceDTO> categoryPreference) {
        this.categoryPreference = categoryPreference;
    }
}