package com.metawebthree.live.application;

import com.metawebthree.live.domain.model.*;
import com.metawebthree.live.domain.repository.*;
import com.metawebthree.live.domain.ports.OrderPort;
import com.metawebthree.live.domain.ports.ProductPort;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

@Service
public class LiveApplicationService {

    private static final Logger logger = LoggerFactory.getLogger(LiveApplicationService.class);

    private final AnchorRepository anchorRepository;
    private final LiveRoomRepository liveRoomRepository;
    private final LiveProductRepository liveProductRepository;
    private final LiveCommentRepository liveCommentRepository;
    private final LiveOrderRepository liveOrderRepository;
    private final OrderPort orderPort;
    private final ProductPort productPort;

    public LiveApplicationService(
            AnchorRepository anchorRepository,
            LiveRoomRepository liveRoomRepository,
            LiveProductRepository liveProductRepository,
            LiveCommentRepository liveCommentRepository,
            LiveOrderRepository liveOrderRepository,
            OrderPort orderPort,
            ProductPort productPort) {
        this.anchorRepository = anchorRepository;
        this.liveRoomRepository = liveRoomRepository;
        this.liveProductRepository = liveProductRepository;
        this.liveCommentRepository = liveCommentRepository;
        this.liveOrderRepository = liveOrderRepository;
        this.orderPort = orderPort;
        this.productPort = productPort;
    }

    public Anchor createAnchor(Long userId, String anchorName, String avatar, String description) {
        Anchor anchor = new Anchor();
        anchor.setUserId(userId);
        anchor.setAnchorName(anchorName);
        anchor.setAvatar(avatar);
        anchor.setDescription(description);
        anchor.setFollowerCount(0);
        anchor.setStatus(1);
        anchor.setCreateTime(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        anchor.setUpdateTime(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        
        Anchor saved = anchorRepository.save(anchor);
        logger.info("Anchor created: {}", saved.getId());
        return saved;
    }

    public LiveRoom startLiveRoom(Long anchorId, String roomName, String coverImage, String description) {
        Anchor anchor = validateAnchor(anchorId);
        LiveRoom room = buildLiveRoom(anchorId, roomName, coverImage, description);
        LiveRoom saved = liveRoomRepository.save(room);
        logger.info("Live room started: {} by anchor: {}", saved.getId(), anchorId);
        return saved;
    }

    private Anchor validateAnchor(Long anchorId) {
        Anchor anchor = anchorRepository.findById(anchorId);
        if (anchor == null) {
            throw new IllegalArgumentException("Anchor not found");
        }
        return anchor;
    }

    private LiveRoom buildLiveRoom(Long anchorId, String roomName, String coverImage, String description) {
        LiveRoom room = new LiveRoom();
        room.setAnchorId(anchorId);
        room.setRoomName(roomName);
        room.setCoverImage(coverImage);
        room.setDescription(description);
        room.setViewerCount(0);
        room.setStatus(1);
        room.setStartTime(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        room.setCreateTime(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        room.setUpdateTime(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        return room;
    }

    public LiveRoom endLiveRoom(Long roomId) {
        LiveRoom room = liveRoomRepository.findById(roomId);
        if (room == null) {
            throw new IllegalArgumentException("Live room not found");
        }

        room.setStatus(2);
        room.setEndTime(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        room.setUpdateTime(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        
        LiveRoom saved = liveRoomRepository.save(room);
        logger.info("Live room ended: {}", roomId);
        return saved;
    }

    public LiveProduct attachProduct(Long roomId, Long productId, Long price, Long discountPrice, Integer stock) {
        validateRoomExists(roomId);
        LiveProduct product = buildLiveProduct(roomId, productId, price, discountPrice, stock);
        LiveProduct saved = liveProductRepository.save(product);
        logger.info("Product attached to room: {} product: {}", roomId, productId);
        return saved;
    }

    private void validateRoomExists(Long roomId) {
        LiveRoom room = liveRoomRepository.findById(roomId);
        if (room == null) {
            throw new IllegalArgumentException("Live room not found");
        }
    }

    private LiveProduct buildLiveProduct(Long roomId, Long productId, Long price, Long discountPrice, Integer stock) {
        LiveProduct product = new LiveProduct();
        product.setRoomId(roomId);
        product.setProductId(productId);
        product.setPrice(price);
        product.setDiscountPrice(discountPrice);
        product.setStock(stock);
        product.setStatus(1);
        product.setCreateTime(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        product.setUpdateTime(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        return product;
    }

    public LiveComment postComment(Long roomId, Long userId, String userName, String content) {
        LiveRoom room = liveRoomRepository.findById(roomId);
        if (room == null) {
            throw new IllegalArgumentException("Live room not found");
        }

        LiveComment comment = new LiveComment();
        comment.setRoomId(roomId);
        comment.setUserId(userId);
        comment.setUserName(userName);
        comment.setContent(content);
        comment.setType(1);
        comment.setCreateTime(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        
        LiveComment saved = liveCommentRepository.save(comment);
        logger.info("Comment posted in room: {} by user: {}", roomId, userId);
        return saved;
    }

    public LiveOrder createLiveOrder(Long roomId, Long productId, Long userId, Integer quantity) {
        LiveProduct product = validateAndGetProduct(productId);
        reduceProductStock(productId, quantity, product);
        Long orderId = createExternalOrder(userId, product.getProductId(), quantity, roomId);
        LiveOrder order = buildLiveOrder(roomId, productId, userId, quantity, orderId, product);
        LiveOrder saved = liveOrderRepository.save(order);
        logger.info("Live order created: {} for user: {} in room: {}", saved.getId(), userId, roomId);
        return saved;
    }

    private LiveProduct validateAndGetProduct(Long productId) {
        LiveProduct product = liveProductRepository.findById(productId);
        if (product == null) {
            throw new IllegalArgumentException("Live product not found");
        }
        return product;
    }

    private void reduceProductStock(Long productId, Integer quantity, LiveProduct product) {
        if (!productPort.reduceStock(product.getProductId(), quantity)) {
            throw new IllegalStateException("Insufficient stock");
        }
    }

    private Long createExternalOrder(Long userId, Long productId, Integer quantity, Long roomId) {
        return orderPort.createOrder(userId, productId, quantity, roomId);
    }

    private LiveOrder buildLiveOrder(Long roomId, Long productId, Long userId, Integer quantity, Long orderId, LiveProduct product) {
        LiveOrder order = new LiveOrder();
        order.setRoomId(roomId);
        order.setProductId(productId);
        order.setUserId(userId);
        order.setOrderId(orderId);
        order.setQuantity(quantity);
        order.setTotalAmount(product.getDiscountPrice() * quantity);
        order.setStatus(1);
        order.setCreateTime(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        order.setUpdateTime(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        return order;
    }

    public Anchor getAnchorById(Long id) {
        return anchorRepository.findById(id);
    }

    public Anchor getAnchorByUserId(Long userId) {
        return anchorRepository.findByUserId(userId);
    }

    public LiveRoom getLiveRoomById(Long id) {
        return liveRoomRepository.findById(id);
    }

    public List<LiveRoom> getLiveRoomsByAnchorId(Long anchorId) {
        return liveRoomRepository.findByAnchorId(anchorId);
    }

    public List<LiveProduct> getLiveProductsByRoomId(Long roomId) {
        return liveProductRepository.findByRoomId(roomId);
    }

    public List<LiveComment> getLiveCommentsByRoomId(Long roomId) {
        return liveCommentRepository.findByRoomId(roomId);
    }

    public List<LiveOrder> getLiveOrdersByRoomId(Long roomId) {
        return liveOrderRepository.findByRoomId(roomId);
    }

    public List<LiveOrder> getLiveOrdersByUserId(Long userId) {
        return liveOrderRepository.findByUserId(userId);
    }
}