from django.contrib import admin
from .models import WorkoutLog


@admin.register(WorkoutLog)
class WorkoutLogAdmin(admin.ModelAdmin):
    list_display = [
        'user', 'date', 'entry_time',
        'exit_time', 'workout_type', 'duration_display'
    ]
    list_filter = ['workout_type', 'date']
    search_fields = ['user__username', 'notes']
    date_hierarchy = 'date'