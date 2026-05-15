package com.metawebthree.cs.domain.model;

import java.time.LocalDateTime;

public class Agent {
    private Long id;
    private Long adminId;
    private String nickname;
    private String avatar;
    private String status;
    private Integer maxConcurrent;
    private Integer currentLoad;
    private Long groupId;
    private String workingHours;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;

    public boolean isAvailable() {
        return "ONLINE".equals(status) && currentLoad != null && maxConcurrent != null && currentLoad < maxConcurrent;
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getAdminId() { return adminId; }
    public void setAdminId(Long adminId) { this.adminId = adminId; }
    public String getNickname() { return nickname; }
    public void setNickname(String nickname) { this.nickname = nickname; }
    public String getAvatar() { return avatar; }
    public void setAvatar(String avatar) { this.avatar = avatar; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public Integer getMaxConcurrent() { return maxConcurrent; }
    public void setMaxConcurrent(Integer maxConcurrent) { this.maxConcurrent = maxConcurrent; }
    public Integer getCurrentLoad() { return currentLoad; }
    public void setCurrentLoad(Integer currentLoad) { this.currentLoad = currentLoad; }
    public Long getGroupId() { return groupId; }
    public void setGroupId(Long groupId) { this.groupId = groupId; }
    public String getWorkingHours() { return workingHours; }
    public void setWorkingHours(String workingHours) { this.workingHours = workingHours; }
    public LocalDateTime getCreateTime() { return createTime; }
    public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
    public LocalDateTime getUpdateTime() { return updateTime; }
    public void setUpdateTime(LocalDateTime updateTime) { this.updateTime = updateTime; }
}
