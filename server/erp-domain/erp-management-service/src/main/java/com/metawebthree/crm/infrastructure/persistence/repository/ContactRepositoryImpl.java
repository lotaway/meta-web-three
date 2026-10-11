package com.metawebthree.crm.infrastructure.persistence.repository;

import com.metawebthree.crm.domain.entity.Contact;
import com.metawebthree.crm.domain.repository.ContactRepository;
import com.metawebthree.crm.infrastructure.persistence.mapper.ContactMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class ContactRepositoryImpl implements ContactRepository {

    private final ContactMapper mapper;

    public ContactRepositoryImpl(ContactMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public Optional<Contact> findById(Long id) {
        return Optional.ofNullable(mapper.selectById(id));
    }

    @Override
    public List<Contact> findAll() {
        return mapper.selectList(null);
    }

    @Override
    public List<Contact> findByCustomerId(Long customerId) {
        return mapper.findByCustomerId(customerId);
    }

    @Override
    public Contact insert(Contact contact) {
        mapper.insert(contact);
        return contact;
    }

    @Override
    public Contact updateById(Contact contact) {
        mapper.updateById(contact);
        return contact;
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }
}
