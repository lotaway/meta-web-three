package com.metawebthree.crm.application.query;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
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
        return contactRepository.selectById(id);
    }

    public List<Contact> listAll() {
        return contactRepository.selectList(null);
    }

    public List<Contact> listByCustomerId(Long customerId) {
        LambdaQueryWrapper<Contact> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Contact::getCustomerId, customerId);
        return contactRepository.selectList(wrapper);
    }
}
