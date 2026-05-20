package com.metawebthree.event;

/**
 * Enumeration of all domain events in the system.
 * Events are organized by domain for clarity.
 */
public enum EventType {
    // Order Domain Events
    ORDER_CREATED("order.created"),
    ORDER_UPDATED("order.updated"),
    ORDER_CANCELLED("order.cancelled"),
    ORDER_COMPLETED("order.completed"),

    // Inventory Domain Events
    INVENTORY_RESERVED("inventory.reserved"),
    INVENTORY_RELEASED("inventory.released"),
    INVENTORY_ADJUSTED("inventory.adjusted"),
    INVENTORY_LOW_STOCK("inventory.low_stock"),

    // Payment Domain Events
    PAYMENT_INITIATED("payment.initiated"),
    PAYMENT_COMPLETED("payment.completed"),
    PAYMENT_FAILED("payment.failed"),
    PAYMENT_REFUNDED("payment.refunded"),

    // Shipment Domain Events
    SHIPMENT_CREATED("shipment.created"),
    SHIPMENT_DISPATCHED("shipment.dispatched"),
    SHIPMENT_DELIVERED("shipment.delivered"),

    // User Domain Events
    USER_REGISTERED("user.registered"),
    USER_UPDATED("user.updated"),

    // Cart Domain Events
    CART_ITEM_ADDED("cart.item_added"),
    CART_ITEM_REMOVED("cart.item_removed"),
    CART_CLEARED("cart.cleared"),

    // Finance Domain Events
    ACCOUNT_CREATED("finance.account_created"),
    ACCOUNT_FROZEN("finance.account_frozen"),
    ACCOUNT_UNFROZEN("finance.account_unfrozen"),
    ACCOUNT_CLOSED("finance.account_closed"),
    ACCOUNT_BALANCE_CHANGED("finance.account_balance_changed"),
    VOUCHER_CREATED("finance.voucher_created"),
    VOUCHER_POSTED("finance.voucher_posted"),
    VOUCHER_REVERSED("finance.voucher_reversed"),

    // Settlement Domain Events
    SETTLEMENT_CREATED("settlement.created"),
    SETTLEMENT_CONFIRMED("settlement.confirmed"),
    SETTLEMENT_COMPLETED("settlement.completed"),
    SETTLEMENT_FAILED("settlement.failed"),
    SETTLEMENT_CANCELLED("settlement.cancelled"),
    RECONCILIATION_COMPLETED("settlement.reconciliation_completed"),

    // Invoice Domain Events
    INVOICE_CREATED("invoice.created"),
    INVOICE_ISSUED("invoice.issued"),
    INVOICE_VOIDED("invoice.voided"),
    INVOICE_RED_FLUSHED("invoice.red_flushed"),

    // Reporting Domain Events
    SALES_REPORT_GENERATED("reporting.sales_report_generated"),
    INVENTORY_REPORT_GENERATED("reporting.inventory_report_generated"),
    FINANCIAL_REPORT_GENERATED("reporting.financial_report_generated"),

    // Procurement Domain Events
    PROCUREMENT_CREATED("procurement.created"),
    PROCUREMENT_APPROVED("procurement.approved"),
    PROCUREMENT_REJECTED("procurement.rejected"),
    PROCUREMENT_COMPLETED("procurement.completed"),
    PROCUREMENT_CANCELLED("procurement.cancelled"),

    // Supplier Domain Events
    SUPPLIER_CREATED("supplier.created"),
    SUPPLIER_UPDATED("supplier.updated"),
    SUPPLIER_ASSESSMENT_CHANGED("supplier.assessment_changed"),
    SUPPLIER_STATUS_CHANGED("supplier.status_changed"),

    // Warehouse Domain Events
    WAREHOUSE_CREATED("warehouse.created"),
    WAREHOUSE_STOCK_IN("warehouse.stock_in"),
    WAREHOUSE_STOCK_OUT("warehouse.stock_out"),
    WAREHOUSE_TRANSFER("warehouse.transfer"),
    INBOUND_ORDER_CREATED("warehouse.inbound_created"),
    INBOUND_ORDER_COMPLETED("warehouse.inbound_completed"),

    // Logistics Domain Events
    LOGISTICS_CREATED("logistics.created"),
    LOGISTICS_TRACKING_UPDATED("logistics.tracking_updated"),
    LOGISTICS_DISPATCHED("logistics.dispatched"),
    LOGISTICS_DELIVERED("logistics.delivered");

    // Reporting Domain Events
    SALES_REPORT_GENERATED("reporting.sales_report_generated"),
    INVENTORY_REPORT_GENERATED("reporting.inventory_report_generated"),
    FINANCIAL_REPORT_GENERATED("reporting.financial_report_generated");

    private final String topic;

    EventType(String topic) {
        this.topic = topic;
    }

    public String getTopic() {
        return topic;
    }
}