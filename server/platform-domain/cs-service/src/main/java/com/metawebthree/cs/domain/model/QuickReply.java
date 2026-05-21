package com.metawebthree.cs.domain.model;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class QuickReply {
    private Long id;
    private Long groupId;
    private String title;
    private String content;
    private String msgType;
    private Integer sort;
    private LocalDateTime createTime;
}
