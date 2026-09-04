package com.metawebthree.email;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Service;

import jakarta.mail.internet.MimeMessage;

@Service
public class EmailSendService {

    private static final Logger logger = LoggerFactory.getLogger(EmailSendService.class);

    private final JavaMailSender mailSender;

    @Value("${notification.email.from:noreply@metawebthree.com}")
    private String fromEmail;

    public EmailSendService(JavaMailSender mailSender) {
        this.mailSender = mailSender;
    }

    public boolean send(String to, String title, String content) {
        try {
            MimeMessage message = mailSender.createMimeMessage();
            MimeMessageHelper helper = new MimeMessageHelper(message, true);
            helper.setFrom(fromEmail);
            helper.setTo(to);
            helper.setSubject(title);
            helper.setText(content, true);
            mailSender.send(message);
            logger.info("Email sent successfully to {}", to);
            return true;
        } catch (Exception e) {
            logger.error("Failed to send email to {}: {}", to, e.getMessage());
            return false;
        }
    }
}
