package com.metawebthree.crm.interfaces.graphql.dto;

import com.metawebthree.crm.domain.entity.CustomerServiceTicket;
import lombok.Data;

@Data
public class TicketDTO implements TypedNode {
    String __typename;
    String id;
    String ticketNo;
    String title;
    String customerId;
    String orderId;
    String type;
    String priority;
    String status;
    String assignedTo;
    String description;
    String resolution;
    String createdAt;
    String updatedAt;

    @Override
    public String __typename() { return __typename; }

    public static TicketDTO from(CustomerServiceTicket ticket) {
        TicketDTO dto = new TicketDTO();
        dto.__typename = "Ticket";
        baseFields(dto, ticket);
        relationFields(dto, ticket);
        ticketFields(dto, ticket);
        timestampFields(dto, ticket);
        return dto;
    }

    private static void baseFields(TicketDTO dto, CustomerServiceTicket ticket) {
        dto.id = ticket.getId().toString();
        dto.ticketNo = ticket.getTicketNo();
        dto.title = ticket.getTitle();
    }

    private static void relationFields(TicketDTO dto, CustomerServiceTicket ticket) {
        dto.customerId = ticket.getCustomerId() != null ? ticket.getCustomerId().toString() : null;
        dto.orderId = ticket.getOrderId() != null ? ticket.getOrderId().toString() : null;
    }

    private static void ticketFields(TicketDTO dto, CustomerServiceTicket ticket) {
        dto.type = ticket.getType();
        dto.priority = ticket.getPriority();
        dto.status = ticket.getStatus();
        dto.assignedTo = ticket.getAssignedTo();
        dto.description = ticket.getDescription();
        dto.resolution = ticket.getResolution();
    }

    private static void timestampFields(TicketDTO dto, CustomerServiceTicket ticket) {
        dto.createdAt = str(ticket.getCreatedAt());
        dto.updatedAt = str(ticket.getUpdatedAt());
    }

    private static String str(Object obj) {
        return obj != null ? obj.toString() : null;
    }
}
