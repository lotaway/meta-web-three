package com.metawebthree.event.domain;

import com.metawebthree.event.BaseEvent;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;

import java.time.Instant;

/**
 * Event published when a shipment is created.
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ShipmentCreatedEvent extends BaseEvent {

    private String shipmentId;
    private String orderId;
    private String carrier;
    private String trackingNumber;
    private String recipientAddress;
    private Instant estimatedDelivery;
}