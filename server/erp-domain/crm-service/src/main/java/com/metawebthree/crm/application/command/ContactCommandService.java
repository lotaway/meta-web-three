package com.metawebthree.crm.application.command;

import com.metawebthree.crm.domain.entity.Contact;
import com.metawebthree.crm.domain.exception.ContactNotFoundException;
import com.metawebthree.crm.domain.repository.ContactRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class ContactCommandService {

    private final ContactRepository contactRepository;

    @Transactional
    public Contact create(Contact contact) {
        contactRepository.insert(contact);
        return contact;
    }

    @Transactional
    public Contact update(Contact contact) {
        Contact existing = contactRepository.selectById(contact.getId());
        if (existing == null) {
            throw new ContactNotFoundException(contact.getId());
        }
        contactRepository.updateById(contact);
        return contact;
    }

    @Transactional
    public void delete(Long id) {
        contactRepository.deleteById(id);
    }
}
