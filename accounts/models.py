from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


class UserProfile(models.Model):
    """Extended user profile storing gym-specific data."""

    GENDER_CHOICES = [
        ('Male', 'Male'),
        ('Female', 'Female'),
    ]

    ACTIVITY_CHOICES = [
        ('sedentary', 'Sedentary (little/no exercise)'),
        ('lightly_active', 'Lightly Active (1-3 days/week)'),
        ('moderately_active', 'Moderately Active (3-5 days/week)'),
        ('very_active', 'Very Active (6-7 days/week)'),
        ('extra_active', 'Extra Active (athlete level)'),
    ]

    PURPOSE_CHOICES = [
        ('weight_loss', 'Weight Loss'),
        ('weight_gain', 'Weight Gain'),
        ('muscle_gain', 'Muscle Gain'),
        ('maintain', 'Maintain Weight'),
    ]

    DIET_CHOICES = [
        ('vegetarian', 'Vegetarian'),
        ('non_vegetarian', 'Non-Vegetarian'),
        ('both', 'Both (Veg & Non-Veg)'),
    ]

    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='profile'
    )
    age = models.PositiveIntegerField(
        null=True, blank=True,
        help_text="Your age in years"
    )
    gender = models.CharField(
        max_length=10,
        choices=GENDER_CHOICES,
        default='Male'
    )
    weight_kg = models.FloatField(
        null=True, blank=True,
        help_text="Weight in kilograms"
    )
    height_cm = models.FloatField(
        null=True, blank=True,
        help_text="Height in centimeters"
    )
    activity_level = models.CharField(
        max_length=20,
        choices=ACTIVITY_CHOICES,
        default='sedentary'
    )
    purpose = models.CharField(
        max_length=20,
        choices=PURPOSE_CHOICES,
        default='maintain'
    )
    diet_preference = models.CharField(
        max_length=20,
        choices=DIET_CHOICES,
        default='both'
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username}'s Profile"

    @property
    def bmi(self):
        """Calculate BMI from weight and height."""
        if self.weight_kg and self.height_cm and self.height_cm > 0:
            return round(
                self.weight_kg / (self.height_cm / 100) ** 2, 2
            )
        return None

    @property
    def bmi_category(self):
        """Return BMI category string."""
        bmi_val = self.bmi
        if bmi_val is None:
            return "N/A"
        if bmi_val < 16:
            return "Severe Thinness"
        elif bmi_val < 17:
            return "Moderate Thinness"
        elif bmi_val < 18.5:
            return "Underweight"
        elif bmi_val < 25:
            return "Normal"
        elif bmi_val < 30:
            return "Overweight"
        elif bmi_val < 35:
            return "Obese I"
        elif bmi_val < 40:
            return "Obese II"
        else:
            return "Obese III"

    @property
    def is_complete(self):
        """Check if all required fields are filled."""
        return all([
            self.age,
            self.gender,
            self.weight_kg,
            self.height_cm,
            self.activity_level,
            self.purpose,
            self.diet_preference,
        ])


# Auto-create profile when User is created
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()