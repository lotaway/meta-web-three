package com.metawebthree.user.domain.model;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Builder
public class MemberAddress {
    private Long id;
    private Long memberId;
    private String name;
    private String phoneNumber;
    private boolean defaultStatus;
    private String postCode;
    private String province;
    private String city;
    private String region;
    private String detailAddress;
}
