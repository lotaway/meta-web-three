package com.metawebthree.crm.application.query;

import com.metawebthree.crm.domain.entity.Contact;
import com.metawebthree.crm.domain.repository.ContactRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class ContactQueryService {

    private final ContactRepository contactRepository;

    public Contact getById(Long id) {
        return contactRepository.findById(id).orElse(null);
    }

    public List<Contact> listAll() {
        return contactRepository.findAll();
    }

    public List<Contact> listByCustomerId(Long customerId) {
        return contactRepository.findByCustomerId(customerId);
    }
}
