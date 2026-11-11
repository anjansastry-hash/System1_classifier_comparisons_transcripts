# Review-gate adjudication sample

20 test errors routed to review by the per-label gate (ModernBERT zero-shot, target 99.5 percent), drawn at random with a fixed seed. Purpose: a routed error only counts as caught if a human reviewer would side with the gold label; if the gold label were wrong, the gate would be sending reviewers to fix the wrong thing. The span column quotes the transcript text that decides the attribute under the benchmark's own present/absent criteria (`BOUNDARIES` in `scripts/jev_noul_benchmark.py`), so any reader can re-check the verdict.

Verdicts: in 20 of 20 rows read, the quoted span supports the gold label. The sample splits into two kinds of error. 9 rows are missed positives: an explicit phrase (an error message, an app fault, an upsell offer) that the model scored below the label's threshold; a reviewer fixes these from the transcript alone. 11 rows are false positives on a definitional boundary: the criteria separate a performance complaint from a service-quality complaint, and a phone-service malfunction from other symptoms, and forbid inferring one from the other. A reviewer who is not given those criteria could reasonably side with the model on these rows, so the gate's value for them depends on the review instructions carrying the criteria, not on the transcript alone.

Blind second read. A second reader then went through the same 20 rows by ear: the transcript and the criterion were read aloud, and the gold label, the model output and the first verdict were withheld until the reader had answered (19 rows; on the remaining row the first verdict was stated first). The first answer matched the gold label on 13 of 19 rows and matched the model on 6. On the 11 definitional-boundary rows the first answer sided with the model on 5, each time reasoning from the overall state of the call rather than from the statement the criterion asks for; once the criterion was applied statement by statement, the reader accepted the gold label on every row (20 of 20 settled as gold supported). Two notes for the criteria themselves: the phone-service criterion lists "calling problems", which can be read to cover "video calls keep lagging"; and a one-line app complaint that the call never returns to was missed by ear, as it was by the model.

| # | Conversation | Attribute | Gold | Model p | Threshold | Margin (nats) | Deciding span | Gold supported | Second reader, blind first answer | Note |
|---:|---:|---|---:|---:|---:|---:|---|---|---|---|
| 1 | 171 | receiving_error_messages | 1 | 0.146 | 0.61 | 2.215 | I keep receiving the message “Service unavailable.” | yes | not blind |  |
| 2 | 380 | poor_service_quality_complaint | 0 | 0.979 | 0.94 | 1.091 | The overall performance has been poor. (names performance; the criteria exclude inferring a service-quality complaint from it) | yes | matched model |  |
| 3 | 191 | poor_performance_complaint | 0 | 0.991 | 0.95 | 1.747 | I am unhappy with the poor service quality. (names service quality; the criteria exclude inferring a performance complaint from it) | yes | matched gold |  |
| 4 | 971 | sales_attempt_by_agent | 1 | 0.050 | 0.05 | 0.008 | I can offer a discounted premium-support package. | yes | matched gold |  |
| 5 | 708 | poor_performance_complaint | 0 | 0.986 | 0.95 | 1.275 | I am unhappy with the poor service quality. (names service quality; the criteria exclude inferring a performance complaint from it) | yes | matched gold |  |
| 6 | 577 | poor_performance_complaint | 0 | 0.984 | 0.95 | 1.178 | I am unhappy with the poor service quality. (names service quality; the criteria exclude inferring a performance complaint from it; glitching is its own attribute) | yes | matched gold |  |
| 7 | 537 | phone_service_issue | 0 | 0.969 | 0.94 | 0.705 | Video calls and online sessions keep lagging. (no mobile or phone service malfunction is reported) | yes | matched model | criterion wording: 'calling problems' can be read to cover 'video calls keep lagging' |
| 8 | 680 | sales_attempt_by_agent | 1 | 0.025 | 0.05 | 0.699 | I can offer a discounted premium-support package. | yes | matched gold |  |
| 9 | 465 | poor_service_quality_complaint | 0 | 0.995 | 0.94 | 2.541 | The overall performance has been poor. (names performance; the criteria exclude inferring a service-quality complaint from it) | yes | matched model |  |
| 10 | 955 | app_issue_mentioned | 1 | 0.128 | 0.13 | 0.020 | The account app is not working correctly. | yes | matched gold |  |
| 11 | 332 | sales_attempt_by_agent | 1 | 0.042 | 0.05 | 0.193 | I can offer a discounted premium-support package. | yes | matched gold |  |
| 12 | 123 | phone_service_issue | 0 | 0.970 | 0.94 | 0.713 | Video calls and online sessions keep lagging. (no mobile or phone service malfunction is reported) | yes | matched gold | criterion wording: 'calling problems' can be read to cover 'video calls keep lagging' |
| 13 | 635 | app_issue_mentioned | 1 | 0.020 | 0.13 | 1.990 | The account app is not working correctly. | yes | matched model | the app sentence is one short line the call never returns to |
| 14 | 171 | poor_service_quality_complaint | 0 | 0.993 | 0.94 | 2.190 | The overall performance has been poor. (names performance; the criteria exclude inferring a service-quality complaint from it) | yes | matched model |  |
| 15 | 591 | poor_service_quality_complaint | 0 | 0.966 | 0.94 | 0.585 | The overall performance has been poor. (names performance; the criteria exclude inferring a service-quality complaint from it) | yes | matched model |  |
| 16 | 956 | poor_service_quality_complaint | 0 | 0.994 | 0.94 | 2.421 | The overall performance has been poor. (names performance; the criteria exclude inferring a service-quality complaint from it; the phone-service line is its own attribute) | yes | matched gold |  |
| 17 | 758 | app_issue_mentioned | 1 | 0.031 | 0.13 | 1.536 | The account app is not working correctly. | yes | matched gold |  |
| 18 | 798 | poor_performance_complaint | 0 | 0.968 | 0.95 | 0.452 | I am unhappy with the poor service quality. (names service quality; the criteria exclude inferring a performance complaint from it) | yes | matched gold |  |
| 19 | 659 | app_issue_mentioned | 1 | 0.017 | 0.13 | 2.151 | The account app is not working correctly. | yes | matched gold |  |
| 20 | 425 | sales_attempt_by_agent | 1 | 0.016 | 0.05 | 1.204 | I can offer a discounted premium-support package. | yes | matched gold |  |

## Transcripts

### 1. Conversation 171 — receiving_error_messages (gold 1, model p 0.146)

```
[Agent]: Welcome to customer care. Please tell me what you need help with.
[Caller]: The overall performance has been poor.
[Agent]: Please describe when the problem happens and which devices are affected.
[Caller]: Video calls and online sessions keep lagging. The download throughput is only about 11 Mbps. I keep receiving the message “Service unavailable.” The problem also occurs outside the building. I cannot connect multiple devices at the same time.
[Agent]: What troubleshooting or previous support has already occurred?
[Caller]: I reset the equipment twice, but the problem continued. This is my second contact about the same issue.
[Agent]: I am taking steps to address this now: I am refreshing the connection profile and reprovisioning the gateway.
[Caller]: That covers everything I needed to report today.
[Agent]: While I have the account open, I can offer a discounted premium-support package.
[Caller]: Please send me the offer details, but do not add anything to the account today.
[Agent]: You’re welcome. Your reference number is SC-0171. Is there anything else I can help with?
```

### 2. Conversation 380 — poor_service_quality_complaint (gold 0, model p 0.979)

```
[Agent]: Hello, you’ve reached customer care. What can I help you with today?
[Caller]: The overall performance has been poor. My phone service is not working correctly.
[Agent]: Please describe when the problem happens and which devices are affected.
[Caller]: Video calls and online sessions keep lagging. I keep receiving the message “Connection failed.” The problem also occurs outside the building.
[Agent]: What troubleshooting or previous support has already occurred?
[Caller]: This is my first contact, and no troubleshooting changes have been made yet.
[Agent]: I am taking steps to address this now: I am refreshing the mobile network profile and account session.
[Caller]: I would like to upgrade my current phone. How long will it take to resolve this?
[Agent]: The current estimated resolution window is 24 to 48 hours. For that upgrade, I can offer a discounted device-protection plan with the new phone.
[Caller]: Please send me the offer details, but do not add anything to the account today.
[Agent]: You’re welcome. Your reference number is SC-0380. Is there anything else I can help with?
```

### 3. Conversation 191 — poor_performance_complaint (gold 0, model p 0.991)

```
[Agent]: Welcome to customer care. Please tell me what you need help with.
[Caller]: I am unhappy with the poor service quality. My phone service is not working correctly. The account app is not working correctly.
[Agent]: Which mobile line and functions are affected?
[Caller]: The problem also occurs outside the building.
[Agent]: What troubleshooting or previous support has already occurred?
[Caller]: This is my fourth contact about the same issue.
[Agent]: I am taking steps to address this now: I am refreshing the mobile network profile and account session.
[Caller]: I would like a credit for the disruption.
[Agent]: I submitted the credit request for billing review.
[Caller]: Thank you for explaining what happens next.
[Agent]: You’re welcome. Your reference number is SC-0191. Is there anything else I can help with?
```

### 4. Conversation 971 — sales_attempt_by_agent (gold 1, model p 0.050)

```
[Agent]: Hi, thank you for calling customer care. How can I help you today?
[Caller]: My phone service is not working correctly.
[Agent]: Which mobile line and functions are affected?
[Caller]: The phone displays “SIM memory full.”
[Agent]: What troubleshooting or previous support has already occurred?
[Caller]: This is my first contact, and no troubleshooting changes have been made yet.
[Agent]: I am taking steps to address this now: I am refreshing the mobile network profile and account session.
[Caller]: I also want to remove the phone line ending in 5302.
[Agent]: I submitted the request to remove the specified phone line. While I have the account open, I can offer a discounted premium-support package.
[Caller]: Please send me the offer details, but do not add anything to the account today.
[Agent]: You’re welcome. Your reference number is SC-0971. Is there anything else I can help with?
```

### 5. Conversation 708 — poor_performance_complaint (gold 0, model p 0.986)

```
[Agent]: Hello, you’ve reached customer care. What can I help you with today?
[Caller]: I am unhappy with the poor service quality. The account app is not working correctly. There is an unfamiliar $35 charge on my bill.
[Agent]: I’ll review the account with you. What other details should I know?
[Caller]: Those are the only symptoms I have observed so far.
[Agent]: What troubleshooting or previous support has already occurred?
[Caller]: This is my first contact, and no troubleshooting changes have been made yet.
[Agent]: I am taking steps to address this now: I am opening a billing investigation and checking the account history.
[Caller]: I would like a credit for the disruption.
[Agent]: I opened a billing investigation for the unfamiliar charge. I submitted the credit request for billing review.
[Caller]: Thank you for explaining what happens next.
[Agent]: You’re welcome. Your reference number is SC-0708. Is there anything else I can help with?
```

### 6. Conversation 577 — poor_performance_complaint (gold 0, model p 0.984)

```
[Agent]: Welcome to customer care. Please tell me what you need help with.
[Caller]: I am unhappy with the poor service quality. The account app is not working correctly. The Smart Home Manager is not controlling my connected devices.
[Agent]: Which connected-home symptoms are you seeing?
[Caller]: The screen keeps glitching and jumping between pages. I keep receiving the message “Profile unavailable.” The problem occurs inside the building.
[Agent]: What troubleshooting or previous support has already occurred?
[Caller]: A technician already visited earlier this week.
[Agent]: I have recorded the symptoms. No troubleshooting changes have been made during this call.
[Caller]: How long will it take to resolve this?
[Agent]: The current estimated resolution window is 24 to 48 hours.
[Caller]: Thank you for explaining what happens next.
[Agent]: You’re welcome. Your reference number is SC-0577. Is there anything else I can help with?
```

### 7. Conversation 537 — phone_service_issue (gold 0, model p 0.969)

```
[Agent]: Thank you for contacting customer care. How may I assist you?
[Caller]: The overall performance has been poor. I am unhappy with the poor service quality.
[Agent]: Please describe when the problem happens and which devices are affected.
[Caller]: Video calls and online sessions keep lagging. The problem occurs inside the building. I cannot connect multiple devices at the same time.
[Agent]: What troubleshooting or previous support has already occurred?
[Caller]: The service problem has left me unable to work today.
[Agent]: I am taking steps to address this now: I am refreshing the connection profile and reprovisioning the gateway.
[Caller]: How long will it take to resolve this?
[Agent]: The current estimated resolution window is 24 to 48 hours.
[Caller]: Thank you for explaining what happens next.
[Agent]: You’re welcome. Your reference number is SC-0537. Is there anything else I can help with?
```

### 8. Conversation 680 — sales_attempt_by_agent (gold 1, model p 0.025)

```
[Agent]: Thank you for contacting customer care. How may I assist you?
[Caller]: The overall performance has been poor. My phone service is not working correctly. I have been unable to activate the internet service.
[Agent]: Please describe when the problem happens and which devices are affected.
[Caller]: Video calls and online sessions keep lagging. I keep receiving the message “Device not found.”
[Agent]: What troubleshooting or previous support has already occurred?
[Caller]: This is my first contact, and no troubleshooting changes have been made yet.
[Agent]: I am taking steps to address this now: I am refreshing the mobile network profile and account session.
[Caller]: I also want to remove the phone line ending in 2914. I would like a credit for the disruption.
[Agent]: I submitted the request to remove the specified phone line. I submitted the credit request for billing review. While I have the account open, I can offer a discounted premium-support package.
[Caller]: Please send me the offer details, but do not add anything to the account today.
[Agent]: You’re welcome. Your reference number is SC-0680. Is there anything else I can help with?
```

### 9. Conversation 465 — poor_service_quality_complaint (gold 0, model p 0.995)

```
[Agent]: Good day, this is customer care. What brings you in today?
[Caller]: The overall performance has been poor.
[Agent]: Please describe when the problem happens and which devices are affected.
[Caller]: The download throughput is only about 8 Mbps. I keep receiving the message “Connection failed.” I cannot connect multiple devices at the same time.
[Agent]: What troubleshooting or previous support has already occurred?
[Caller]: I reset the equipment twice, but the problem continued. A technician already visited earlier this week. This is my second contact about the same issue. The service problem has left me unable to work today.
[Agent]: I have recorded the symptoms. No troubleshooting changes have been made during this call.
[Caller]: That covers everything I needed to report today.
[Agent]: I have added detailed notes so the next support team can continue from here.
[Caller]: Thank you for explaining what happens next.
[Agent]: You’re welcome. Your reference number is SC-0465. Is there anything else I can help with?
```

### 10. Conversation 955 — app_issue_mentioned (gold 1, model p 0.128)

```
[Agent]: Welcome to customer care. Please tell me what you need help with.
[Caller]: I am unhappy with the poor service quality. The account app is not working correctly. The Smart Home Manager is not controlling my connected devices.
[Agent]: Which connected-home symptoms are you seeing?
[Caller]: The screen keeps glitching and jumping between pages. The problem occurs inside the building. The problem also occurs outside the building. I cannot connect multiple devices at the same time.
[Agent]: What troubleshooting or previous support has already occurred?
[Caller]: This is my first contact, and no troubleshooting changes have been made yet.
[Agent]: I am taking steps to address this now: I am resynchronizing the Smart Home Manager profile and connected-device permissions.
[Caller]: I would like a credit for the disruption. How long will it take to resolve this?
[Agent]: I submitted the credit request for billing review. The current estimated resolution window is one business day.
[Caller]: Thank you for explaining what happens next.
[Agent]: You’re welcome. Your reference number is SC-0955. Is there anything else I can help with?
```

### 11. Conversation 332 — sales_attempt_by_agent (gold 1, model p 0.042)

```
[Agent]: Hi, thank you for calling customer care. How can I help you today?
[Caller]: The overall performance has been poor.
[Agent]: Please describe when the problem happens and which devices are affected.
[Caller]: The screen keeps glitching and jumping between pages. The problem occurs inside the building.
[Agent]: What troubleshooting or previous support has already occurred?
[Caller]: The service problem has left me unable to work today.
[Agent]: I am taking steps to address this now: I am refreshing the connection profile and reprovisioning the gateway.
[Caller]: I would like a credit for the disruption.
[Agent]: I submitted the credit request for billing review. While I have the account open, I can offer a discounted premium-support package.
[Caller]: Please send me the offer details, but do not add anything to the account today.
[Agent]: You’re welcome. Your reference number is SC-0332. Is there anything else I can help with?
```

### 12. Conversation 123 — phone_service_issue (gold 0, model p 0.970)

```
[Agent]: Hi, thank you for calling customer care. How can I help you today?
[Caller]: The overall performance has been poor. I am unhappy with the poor service quality.
[Agent]: Please describe when the problem happens and which devices are affected.
[Caller]: Video calls and online sessions keep lagging. I keep receiving the message “Connection failed.” The problem occurs inside the building. The problem also occurs outside the building.
[Agent]: What troubleshooting or previous support has already occurred?
[Caller]: The service problem has left me unable to work today.
[Agent]: I am taking steps to address this now: I am refreshing the connection profile and reprovisioning the gateway.
[Caller]: That covers everything I needed to report today.
[Agent]: I have added detailed notes so the next support team can continue from here.
[Caller]: Thank you for explaining what happens next.
[Agent]: You’re welcome. Your reference number is SC-0123. Is there anything else I can help with?
```

### 13. Conversation 635 — app_issue_mentioned (gold 1, model p 0.020)

```
[Agent]: Hi, thank you for calling customer care. How can I help you today?
[Caller]: My phone service is not working correctly. The account app is not working correctly.
[Agent]: Which mobile line and functions are affected?
[Caller]: The problem occurs inside the building.
[Agent]: What troubleshooting or previous support has already occurred?
[Caller]: This is my first contact, and no troubleshooting changes have been made yet.
[Agent]: I am taking steps to address this now: I am refreshing the mobile network profile and account session.
[Caller]: I would like to upgrade my current phone.
[Agent]: I documented the phone-upgrade request for the device team.
[Caller]: Thank you for explaining what happens next.
[Agent]: You’re welcome. Your reference number is SC-0635. Is there anything else I can help with?
```

### 14. Conversation 171 — poor_service_quality_complaint (gold 0, model p 0.993)

```
[Agent]: Welcome to customer care. Please tell me what you need help with.
[Caller]: The overall performance has been poor.
[Agent]: Please describe when the problem happens and which devices are affected.
[Caller]: Video calls and online sessions keep lagging. The download throughput is only about 11 Mbps. I keep receiving the message “Service unavailable.” The problem also occurs outside the building. I cannot connect multiple devices at the same time.
[Agent]: What troubleshooting or previous support has already occurred?
[Caller]: I reset the equipment twice, but the problem continued. This is my second contact about the same issue.
[Agent]: I am taking steps to address this now: I am refreshing the connection profile and reprovisioning the gateway.
[Caller]: That covers everything I needed to report today.
[Agent]: While I have the account open, I can offer a discounted premium-support package.
[Caller]: Please send me the offer details, but do not add anything to the account today.
[Agent]: You’re welcome. Your reference number is SC-0171. Is there anything else I can help with?
```

### 15. Conversation 591 — poor_service_quality_complaint (gold 0, model p 0.966)

```
[Agent]: Hi, thank you for calling customer care. How can I help you today?
[Caller]: The overall performance has been poor.
[Agent]: Please describe when the problem happens and which devices are affected.
[Caller]: Video calls and online sessions keep lagging. The download throughput is only about 6 Mbps. I keep receiving the message “Device not found.” The problem occurs inside the building.
[Agent]: What troubleshooting or previous support has already occurred?
[Caller]: A technician already visited earlier this week.
[Agent]: I am taking steps to address this now: I am refreshing the connection profile and reprovisioning the gateway.
[Caller]: That covers everything I needed to report today.
[Agent]: I have added detailed notes so the next support team can continue from here.
[Caller]: Thank you for explaining what happens next.
[Agent]: You’re welcome. Your reference number is SC-0591. Is there anything else I can help with?
```

### 16. Conversation 956 — poor_service_quality_complaint (gold 0, model p 0.994)

```
[Agent]: Thank you for contacting customer care. How may I assist you?
[Caller]: The overall performance has been poor. My phone service is not working correctly.
[Agent]: Please describe when the problem happens and which devices are affected.
[Caller]: The download throughput is only about 6 Mbps. The problem occurs inside the building.
[Agent]: What troubleshooting or previous support has already occurred?
[Caller]: This is my first contact, and no troubleshooting changes have been made yet.
[Agent]: I am taking steps to address this now: I am refreshing the mobile network profile and account session.
[Caller]: That covers everything I needed to report today.
[Agent]: I have added detailed notes so the next support team can continue from here.
[Caller]: Thank you for explaining what happens next.
[Agent]: You’re welcome. Your reference number is SC-0956. Is there anything else I can help with?
```

### 17. Conversation 758 — app_issue_mentioned (gold 1, model p 0.031)

```
[Agent]: Good day, this is customer care. What brings you in today?
[Caller]: The overall performance has been poor. My phone service is not working correctly. The account app is not working correctly.
[Agent]: Please describe when the problem happens and which devices are affected.
[Caller]: Video calls and online sessions keep lagging. I cannot connect multiple devices at the same time.
[Agent]: What troubleshooting or previous support has already occurred?
[Caller]: This is my third contact about the same issue.
[Agent]: I am taking steps to address this now: I am refreshing the mobile network profile and account session.
[Caller]: That covers everything I needed to report today.
[Agent]: I have added detailed notes so the next support team can continue from here.
[Caller]: Thank you for explaining what happens next.
[Agent]: You’re welcome. Your reference number is SC-0758. Is there anything else I can help with?
```

### 18. Conversation 798 — poor_performance_complaint (gold 0, model p 0.968)

```
[Agent]: Thank you for contacting customer care. How may I assist you?
[Caller]: I am unhappy with the poor service quality. There is an unfamiliar $64 charge on my bill.
[Agent]: I’ll review the account with you. What other details should I know?
[Caller]: Those are the only symptoms I have observed so far.
[Agent]: What troubleshooting or previous support has already occurred?
[Caller]: This is my first contact, and no troubleshooting changes have been made yet.
[Agent]: I am taking steps to address this now: I am opening a billing investigation and checking the account history.
[Caller]: I would like a credit for the disruption.
[Agent]: I opened a billing investigation for the unfamiliar charge. I submitted the credit request for billing review.
[Caller]: Thank you for explaining what happens next.
[Agent]: You’re welcome. Your reference number is SC-0798. Is there anything else I can help with?
```

### 19. Conversation 659 — app_issue_mentioned (gold 1, model p 0.017)

```
[Agent]: Good day, this is customer care. What brings you in today?
[Caller]: I am unhappy with the poor service quality. My phone service is not working correctly. The account app is not working correctly.
[Agent]: Which mobile line and functions are affected?
[Caller]: Those are the only symptoms I have observed so far.
[Agent]: What troubleshooting or previous support has already occurred?
[Caller]: This is my first contact, and no troubleshooting changes have been made yet.
[Agent]: I am taking steps to address this now: I am refreshing the mobile network profile and account session.
[Caller]: I would like to upgrade my current phone.
[Agent]: I documented the phone-upgrade request for the device team.
[Caller]: Thank you for explaining what happens next.
[Agent]: You’re welcome. Your reference number is SC-0659. Is there anything else I can help with?
```

### 20. Conversation 425 — sales_attempt_by_agent (gold 1, model p 0.016)

```
[Agent]: Welcome to customer care. Please tell me what you need help with.
[Caller]: My phone service is not working correctly.
[Agent]: Which mobile line and functions are affected?
[Caller]: Those are the only symptoms I have observed so far.
[Agent]: What troubleshooting or previous support has already occurred?
[Caller]: This is my first contact, and no troubleshooting changes have been made yet.
[Agent]: I am taking steps to address this now: I am refreshing the mobile network profile and account session.
[Caller]: I also want to remove the phone line ending in 8264. I would like a credit for the disruption.
[Agent]: I submitted the request to remove the specified phone line. I have applied a $20 credit to the account. While I have the account open, I can offer a discounted premium-support package.
[Caller]: Please send me the offer details, but do not add anything to the account today.
[Agent]: You’re welcome. Your reference number is SC-0425. Is there anything else I can help with?
```

Generated by `scripts/review_gate_economics.py`.
