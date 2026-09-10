from django.db import models
from django.conf import settings


class WorkoutLog(models.Model):
    WORKOUT_CHOICES = [
        ('cardio', 'Cardio'),
        ('strength', 'Strength Training'),
        ('hiit', 'HIIT'),
        ('yoga', 'Yoga'),
        ('crossfit', 'CrossFit'),
        ('pilates', 'Pilates'),
        ('cycling', 'Cycling'),
        ('swimming', 'Swimming'),
        ('running', 'Running'),
        ('stretching', 'Stretching'),
        ('martial_arts', 'Martial Arts'),
        ('other', 'Other'),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='workout_logs'
    )
    date = models.DateField()
    entry_time = models.TimeField()
    exit_time = models.TimeField()
    workout_type = models.CharField(
        max_length=20,
        choices=WORKOUT_CHOICES
    )
    notes = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-date', '-entry_time']

    def __str__(self):
        return (
            f"{self.user.username} — {self.get_workout_type_display()} "
            f"on {self.date}"
        )

    @property
    def duration_minutes(self):
        """Calculate workout duration in minutes."""
        from datetime import datetime, timedelta

        entry = datetime.combine(self.date, self.entry_time)
        exit_ = datetime.combine(self.date, self.exit_time)

        # Handle case where exit is past midnight
        if exit_ < entry:
            exit_ += timedelta(days=1)

        diff = exit_ - entry
        return int(diff.total_seconds() // 60)

    @property
    def duration_display(self):
        """Return formatted duration string."""
        mins = self.duration_minutes
        hours = mins // 60
        remaining = mins % 60
        if hours > 0:
            return f"{hours}h {remaining}m"
        return f"{remaining}m"