package com.metawebthree.user.application.dto;

import lombok.Data;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NoArgsConstructor;

import java.util.Date;
import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SubTokenDTO {
    private String token;
    private Date expiresAt;
    private List<String> permissions;
    private String parentToken;
}