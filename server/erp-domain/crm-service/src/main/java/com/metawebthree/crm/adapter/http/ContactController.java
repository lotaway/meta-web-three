package com.metawebthree.crm.adapter.http;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.crm.adapter.vo.Result;
import com.metawebthree.crm.domain.entity.Contact;
import com.metawebthree.crm.domain.exception.ContactNotFoundException;
import com.metawebthree.crm.domain.repository.ContactRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/crm/contacts")
@RequiredArgsConstructor
public class ContactController {

    private final ContactRepository contactRepository;

    @GetMapping("/{id}")
    public Result<Contact> getById(@PathVariable Long id) {
        return Result.success(contactRepository.selectById(id));
    }

    @GetMapping("/list")
    public Result<List<Contact>> listAll() {
        return Result.success(contactRepository.selectList(null));
    }

    @GetMapping("/customer/{customerId}")
    public Result<List<Contact>> getByCustomer(@PathVariable Long customerId) {
        LambdaQueryWrapper<Contact> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Contact::getCustomerId, customerId);
        return Result.success(contactRepository.selectList(wrapper));
    }

    @PostMapping
    public Result<Contact> create(@RequestBody Contact contact) {
        contactRepository.insert(contact);
        return Result.success(contact);
    }

    @PutMapping
    public Result<Contact> update(@RequestBody Contact contact) {
        Contact existing = contactRepository.selectById(contact.getId());
        if (existing == null) {
            throw new ContactNotFoundException(contact.getId());
        }
        contactRepository.updateById(contact);
        return Result.success(contact);
    }

    @DeleteMapping("/{id}")
    public Result<Void> delete(@PathVariable Long id) {
        contactRepository.deleteById(id);
        return Result.success();
    }
}
