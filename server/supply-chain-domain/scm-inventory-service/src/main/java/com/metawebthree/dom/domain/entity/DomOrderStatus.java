package com.metawebthree.dom.domain.entity;

public enum DomOrderStatus {
    PENDING,
    ATP_CHECKING,
    ATP_FAILED,
    SOURCING,
    SOURCING_FAILED,
    SOURCING_COMPLETED,
    PARTIALLY_FULFILLED,
    FULFILLED,
    CANCELLED
}
