from django.contrib import admin
from .models import UserProfile


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = [
        'user', 'age', 'gender', 'weight_kg',
        'height_cm', 'activity_level', 'purpose',
        'diet_preference', 'bmi',
    ]
    list_filter = [
        'gender', 'activity_level', 'purpose',
        'diet_preference',
    ]
    search_fields = [
        'user__username', 'user__first_name',
        'user__last_name',
    ]
    readonly_fields = ['created_at', 'updated_at']