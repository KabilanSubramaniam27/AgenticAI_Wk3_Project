# SeniorCare Synthetic Data

All datasets are synthetic and intended for demo/testing only.

## Shared ID relationships
- seniors.json: seniorId is the main entity key.
- caregivers.json -> seniorId
- providers.json -> providerId
- provider_availability.json -> providerId
- appointments.json -> seniorId, providerId, referralId, transportationRideId
- referrals.json -> seniorId, referringProviderId, referredProviderId, appointmentId
- medications.json -> seniorId, prescribingProviderId
- pharmacy_refills.json -> seniorId, medicationId
- transportation_services.json -> transportationServiceId
- transportation_vehicles.json -> vehicleId, transportationServiceId
- rides.json -> seniorId, appointmentId, transportationServiceId, vehicleId
- discharge_tasks.json -> seniorId, relatedAppointmentId, relatedMedicationId
- meal_services.json -> mealServiceId
- meal_enrollments.json -> seniorId, recipientId, mealServiceId
- benefit_applications.json -> seniorId
- home_support_requests.json -> seniorId
- social_activities.json -> activityId
- activity_registrations.json -> seniorId, recipientId, activityId
- case_tasks.json -> seniorId, assignedCaregiverId, relatedEntityId
- reminders.json -> seniorId, caregiverId, relatedEntityId

Seed counts vary by dataset, and local simulation tools append operational records during use.
These files are transactional demo state and are deliberately kept out of the Actian public-knowledge vector corpus.
