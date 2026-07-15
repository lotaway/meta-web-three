package com.metawebthree.crm.domain.repository;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.crm.domain.entity.Contact;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ContactRepository extends BaseMapper<Contact> {
}
