package com.metawebthree.crm.interfaces.graphql.dto;

import com.metawebthree.crm.domain.entity.Contact;
import lombok.Data;

@Data
public class ContactDTO implements TypedNode {
    String __typename;
    String id;
    String firstName;
    String lastName;
    String email;
    String phone;
    String mobile;
    String position;
    String department;
    String customerId;
    Boolean isPrimary;
    String city;
    String createdAt;
    String updatedAt;

    @Override
    public String __typename() { return __typename; }

    public static ContactDTO from(Contact contact) {
        ContactDTO dto = new ContactDTO();
        dto.__typename = "Contact";
        baseFields(dto, contact);
        orgFields(dto, contact);
        locationFields(dto, contact);
        timestampFields(dto, contact);
        return dto;
    }

    private static void baseFields(ContactDTO dto, Contact contact) {
        dto.id = contact.getId().toString();
        dto.firstName = contact.getFirstName();
        dto.lastName = contact.getLastName();
        dto.email = contact.getEmail();
        dto.phone = contact.getPhone();
        dto.mobile = contact.getMobile();
    }

    private static void orgFields(ContactDTO dto, Contact contact) {
        dto.position = contact.getPosition();
        dto.department = contact.getDepartment();
        dto.customerId = contact.getCustomerId() != null ? contact.getCustomerId().toString() : null;
        dto.isPrimary = contact.getIsPrimary();
    }

    private static void locationFields(ContactDTO dto, Contact contact) {
        dto.city = contact.getCity();
    }

    private static void timestampFields(ContactDTO dto, Contact contact) {
        dto.createdAt = str(contact.getCreatedAt());
        dto.updatedAt = str(contact.getUpdatedAt());
    }

    private static String str(Object obj) {
        return obj != null ? obj.toString() : null;
    }
}
