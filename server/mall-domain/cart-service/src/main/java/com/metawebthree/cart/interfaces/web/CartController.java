package com.metawebthree.cart.interfaces.web;

import com.metawebthree.cart.application.CartService;
import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.cart.dto.CartItemDTO;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Validated
@RestController
@RequestMapping("/cart")
@RequiredArgsConstructor
@Tag(name = "Cart Management", description = "Endpoints for shopping cart lifecycle and management")
public class CartController {

    private final CartService cartService;

    @Operation(summary = "Add item to cart")
    @PostMapping("/add")
    public ApiResponse<Integer> add(@RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestBody CartItemDTO cartItem) {
        cartItem.setMemberId(userId);
        return ApiResponse.success(cartService.add(cartItem));
    }

    @Operation(summary = "List current user's cart items")
    @GetMapping("/list")
    public ApiResponse<List<CartItemDTO>> list(@RequestHeader(HeaderConstants.USER_ID) Long userId) {
        return ApiResponse.success(cartService.list(userId));
    }

    @Operation(summary = "Update cart item quantity")
    @GetMapping("/update/quantity")
    public ApiResponse<Integer> updateQuantity(@RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestParam Long id,
            @RequestParam Integer quantity) {
        return ApiResponse.success(cartService.updateQuantity(userId, id, quantity));
    }

    @Operation(summary = "Remove items from cart")
    @PostMapping("/delete")
    public ApiResponse<Integer> delete(@RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestParam("ids") List<Long> ids) {
        return ApiResponse.success(cartService.delete(userId, ids));
    }

    @Operation(summary = "Clear current user's cart")
    @PostMapping("/clear")
    public ApiResponse<Integer> clear(@RequestHeader(HeaderConstants.USER_ID) Long userId) {
        return ApiResponse.success(cartService.clear(userId));
    }

    @Operation(summary = "List cart items with promotion info")
    @GetMapping("/list/promotion")
    public ApiResponse<List<CartItemDTO>> listWithPromotion(@RequestHeader(HeaderConstants.USER_ID) Long userId) {
        return ApiResponse.success(cartService.listWithPromotion(userId));
    }

    @Operation(summary = "Update cart item attributes/SKU")
    @PostMapping("/update/attr")
    public ApiResponse<Void> updateAttributes(@RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestParam Long id,
            @RequestBody CartItemDTO cartItem) {
        cartService.updateAttributes(userId, id, cartItem);
        return ApiResponse.success();
    }

    @Operation(summary = "Get product SKU options for cart")
    @GetMapping("/getProduct/{productId}")
    public ApiResponse<CartItemDTO> getProduct(@RequestHeader(HeaderConstants.USER_ID) Long userId,
            @PathVariable Long productId) {
        return ApiResponse.success(cartService.getProductOptions(userId, productId));
    }
}
