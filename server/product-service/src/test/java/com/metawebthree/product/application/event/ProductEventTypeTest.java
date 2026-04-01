package com.metawebthree.product.application.event;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class ProductEventTypeTest {

    @Test
    void productDeleted_shouldMatchExpectedEventName() {
        assertEquals("product.deleted", ProductEventType.PRODUCT_DELETED.getEventName());
    }
}
