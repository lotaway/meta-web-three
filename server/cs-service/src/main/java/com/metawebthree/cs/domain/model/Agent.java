package com.metawebthree.cs.domain.model;

import lombok.Data;

import java.time.LocalDateTime;

@Data
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
}
