package com.metawebthree.procurement.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.procurement.infrastructure.persistence.dataobject.ProcurementReturnOrderItemDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import java.util.List;

@Mapper
public interface ProcurementReturnOrderItemMapper extends BaseMapper<ProcurementReturnOrderItemDO> {
    
    List<ProcurementReturnOrderItemDO> selectByReturnOrderId(@Param("returnOrderId") Long returnOrderId);
    
    List<ProcurementReturnOrderItemDO> selectByReturnNo(@Param("returnNo") String returnNo);
}