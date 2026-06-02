package com.metawebthree.warehouse.application;

import com.metawebthree.warehouse.application.dto.StocktakeOrderDTO;
import java.util.List;

public interface StocktakeOrderService {
    StocktakeOrderDTO createStocktakeOrder(StocktakeOrderDTO dto);
    StocktakeOrderDTO submitStocktakeOrder(String orderNo);
    StocktakeOrderDTO startStocktake(String orderNo);
    StocktakeOrderDTO completeCounting(String orderNo);
    StocktakeOrderDTO reportDiscrepancy(String orderNo);
    StocktakeOrderDTO adjustInventory(String orderNo);
    StocktakeOrderDTO completeStocktake(String orderNo);
    StocktakeOrderDTO cancelStocktake(String orderNo);
    StocktakeOrderDTO queryStocktakeOrder(String orderNo);
    List<StocktakeOrderDTO> listStocktakeOrders(Long warehouseId, String status);
}