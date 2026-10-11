package com.metawebthree.crm.domain.repository;

import com.metawebthree.crm.domain.entity.Contact;

import java.util.List;
import java.util.Optional;

public interface ContactRepository {
    Optional<Contact> findById(Long id);
    List<Contact> findAll();
    List<Contact> findByCustomerId(Long customerId);
    Contact insert(Contact contact);
    Contact updateById(Contact contact);
    void deleteById(Long id);
}
