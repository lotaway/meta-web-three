package com.metawebthree.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.pojo.AuthorPojo;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface AuthorMapper extends BaseMapper<AuthorPojo> {

}
