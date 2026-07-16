package com.metawebthree.event;

public enum EventType {

    ORDER_CREATED("order.created"),
    ORDER_UPDATED("order.updated"),
    ORDER_CANCELLED("order.cancelled"),
    ORDER_COMPLETED("order.completed"),

    INVENTORY_RESERVED("inventory.reserved"),
    INVENTORY_RELEASED("inventory.released"),
    INVENTORY_ADJUSTED("inventory.adjusted"),
    INVENTORY_LOW_STOCK("inventory.low_stock"),

    PAYMENT_INITIATED("payment.initiated"),
    PAYMENT_COMPLETED("payment.completed"),
    PAYMENT_FAILED("payment.failed"),
    PAYMENT_REFUNDED("payment.refunded"),

    SHIPMENT_CREATED("shipment.created"),
    SHIPMENT_DISPATCHED("shipment.dispatched"),
    SHIPMENT_DELIVERED("shipment.delivered"),

    USER_REGISTERED("user.registered"),
    USER_UPDATED("user.updated"),

    CART_ITEM_ADDED("cart.item_added"),
    CART_ITEM_REMOVED("cart.item_removed"),
    CART_CLEARED("cart.cleared"),

    ACCOUNT_CREATED("finance.account_created"),
    ACCOUNT_FROZEN("finance.account_frozen"),
    ACCOUNT_UNFROZEN("finance.account_unfrozen"),
    ACCOUNT_CLOSED("finance.account_closed"),
    ACCOUNT_BALANCE_CHANGED("finance.account_balance_changed"),
    VOUCHER_CREATED("finance.voucher_created"),
    VOUCHER_POSTED("finance.voucher_posted"),
    VOUCHER_REVERSED("finance.voucher_reversed"),

    SETTLEMENT_CREATED("settlement.created"),
    SETTLEMENT_CONFIRMED("settlement.confirmed"),
    SETTLEMENT_COMPLETED("settlement.completed"),
    SETTLEMENT_FAILED("settlement.failed"),
    SETTLEMENT_CANCELLED("settlement.cancelled"),
    RECONCILIATION_COMPLETED("settlement.reconciliation_completed"),

    INVOICE_CREATED("invoice.created"),
    INVOICE_ISSUED("invoice.issued"),
    INVOICE_VOIDED("invoice.voided"),
    INVOICE_RED_FLUSHED("invoice.red_flushed"),

    SALES_REPORT_GENERATED("reporting.sales_report_generated"),
    INVENTORY_REPORT_GENERATED("reporting.inventory_report_generated"),
    FINANCIAL_REPORT_GENERATED("reporting.financial_report_generated"),

    PROCUREMENT_CREATED("procurement.created"),
    PROCUREMENT_APPROVED("procurement.approved"),
    PROCUREMENT_REJECTED("procurement.rejected"),
    PROCUREMENT_COMPLETED("procurement.completed"),
    PROCUREMENT_CANCELLED("procurement.cancelled"),

    SUPPLIER_CREATED("supplier.created"),
    SUPPLIER_UPDATED("supplier.updated"),
    SUPPLIER_ASSESSMENT_CHANGED("supplier.assessment_changed"),
    SUPPLIER_STATUS_CHANGED("supplier.status_changed"),

    WAREHOUSE_CREATED("warehouse.created"),
    WAREHOUSE_STOCK_IN("warehouse.stock_in"),
    WAREHOUSE_STOCK_OUT("warehouse.stock_out"),
    WAREHOUSE_TRANSFER("warehouse.transfer"),
    INBOUND_ORDER_CREATED("warehouse.inbound_created"),
    INBOUND_ORDER_COMPLETED("warehouse.inbound_completed"),

    LOGISTICS_CREATED("logistics.created"),
    LOGISTICS_TRACKING_UPDATED("logistics.tracking_updated"),
    LOGISTICS_DISPATCHED("logistics.dispatched"),
    LOGISTICS_DELIVERED("logistics.delivered"),

    PRODUCTION_ORDER_RELEASED("production.order_released"),
    MES_WORK_ORDER_COMPLETED("mes.work_order_completed"),
    MES_WORK_ORDER_STARTED("mes.work_order_started"),
    MES_TASK_COMPLETED("mes.task_completed"),
    COST_ACCOUNTING_TRIGGERED("finance.cost_accounting_triggered");

    public static final String MES_WORK_ORDER_COMPLETED_TOPIC = "mes.work_order_completed";
    public static final String MES_TASK_COMPLETED_TOPIC = "mes.task_completed";
    public static final String PRODUCTION_EVENTS_TOPIC = "production.events";

    private final String topic;

    EventType(String topic) {
        this.topic = topic;
    }

    public String getTopic() {
        return topic;
    }
}
