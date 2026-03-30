package com.metawebthree.order.domain.model;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class OrderSetting {
    private final Long id;
    private final Integer flashOrderOvertime;
    private final Integer normalOrderOvertime;
    private final Integer confirmOvertime;
    private final Integer finishOvertime;
    private final Integer commentOvertime;
}
