package com.metawebthree.crm.domain.repository;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.crm.domain.entity.Contact;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface ContactRepository extends BaseMapper<Contact> {

    @Select("SELECT * FROM crm_contact WHERE customer_id = #{customerId} AND deleted = 0")
    List<Contact> findByCustomerId(@Param("customerId") Long customerId);
}
