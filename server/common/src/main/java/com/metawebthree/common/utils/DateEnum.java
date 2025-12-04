package com.metawebthree.common.utils;

import java.util.Date;

public enum DateEnum {
    ONE_SECOND(1000L),
    ONE_MINUTE(60L * 1000),
    ONE_HOUR(60L * 60 * 1000),
    ONE_DAY(24L * 60 * 60 * 1000),
    ONE_WEEK(7L * 24 * 60 * 60 * 1000),
    ONE_MONTH(30L * 24 * 60 * 60 * 1000),
    ONE_HALF_YEAR(180L * 24 * 60 * 60 * 1000),
    ONE_YEAR(365L * 24 * 60 * 60 * 1000),
    ONE_HUNDRED_YEAR(100L * 365 * 24 * 60 * 60 * 1000);

    private final Long value;

    DateEnum(Long l) {
        this.value = l;
    }

    public Long getValue() {
        return value;
    }

    public Long toAfterThis() {
        return System.currentTimeMillis() + this.value;
    }

    public Date toAfterThisAsDate() {
        return new Date(this.toAfterThis());
    }

    public Long toBeforeThis() {
        return System.currentTimeMillis() - this.value;
    }

    public Date toBeforeThisAsDate() {
        return new Date(this.toBeforeThis());
    }
}
