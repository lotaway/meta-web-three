package com.metawebthree.user.domain.model;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class MemberAddress {
    private final Long id;
    private final Long memberId;
    private final String name;
    private final String phoneNumber;
    private final boolean defaultStatus;
    private final String postCode;
    private final String province;
    private final String city;
    private final String region;
    private final String detailAddress;
}
