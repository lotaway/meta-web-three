package com.metawebthree.dataanalysis.application.dto;

public class UserProfileDTO {
    private Long userId;
    private Integer age;
    private String gender;
    private String region;
    private String preference;
    private Integer purchaseFrequency;
    private Long totalSpent;
    private Integer orderCount;
    private String categoryPreference;

    public Long getUserId() {
        return userId;
    }

    public void setUserId(Long userId) {
        this.userId = userId;
    }

    public Integer getAge() {
        return age;
    }

    public void setAge(Integer age) {
        this.age = age;
    }

    public String getGender() {
        return gender;
    }

    public void setGender(String gender) {
        this.gender = gender;
    }

    public String getRegion() {
        return region;
    }

    public void setRegion(String region) {
        this.region = region;
    }

    public String getPreference() {
        return preference;
    }

    public void setPreference(String preference) {
        this.preference = preference;
    }

    public Integer getPurchaseFrequency() {
        return purchaseFrequency;
    }

    public void setPurchaseFrequency(Integer purchaseFrequency) {
        this.purchaseFrequency = purchaseFrequency;
    }

    public Long getTotalSpent() {
        return totalSpent;
    }

    public void setTotalSpent(Long totalSpent) {
        this.totalSpent = totalSpent;
    }

    public Integer getOrderCount() {
        return orderCount;
    }

    public void setOrderCount(Integer orderCount) {
        this.orderCount = orderCount;
    }

    public String getCategoryPreference() {
        return categoryPreference;
    }

    public void setCategoryPreference(String categoryPreference) {
        this.categoryPreference = categoryPreference;
    }
}