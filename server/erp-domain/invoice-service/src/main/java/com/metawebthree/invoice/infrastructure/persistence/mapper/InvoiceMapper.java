package com.metawebthree.invoice.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.invoice.infrastructure.persistence.dataobject.InvoiceDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface InvoiceMapper extends BaseMapper<InvoiceDO> {
}