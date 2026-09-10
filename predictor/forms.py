from django import forms
from .models import WorkoutLog
from datetime import date, datetime


class WorkoutLogForm(forms.ModelForm):
    date = forms.DateField(
        widget=forms.DateInput(attrs={
            'type': 'date',
            'class': 'form-control bg-dark text-white border-secondary',
            'max': date.today().isoformat(),
        })
    )
    entry_time = forms.TimeField(
        widget=forms.TimeInput(attrs={
            'type': 'time',
            'class': 'form-control bg-dark text-white border-secondary',
        })
    )
    exit_time = forms.TimeField(
        widget=forms.TimeInput(attrs={
            'type': 'time',
            'class': 'form-control bg-dark text-white border-secondary',
        })
    )
    workout_type = forms.ChoiceField(
        choices=WorkoutLog.WORKOUT_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-select bg-dark text-white border-secondary',
        })
    )
    notes = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control bg-dark text-white border-secondary',
            'rows': 3,
            'placeholder': 'Optional notes about your workout...',
        })
    )

    class Meta:
        model = WorkoutLog
        fields = ['date', 'entry_time', 'exit_time',
                  'workout_type', 'notes']

    def clean(self):
        cleaned_data = super().clean()
        entry = cleaned_data.get('entry_time')
        exit_ = cleaned_data.get('exit_time')
        log_date = cleaned_data.get('date')

        if log_date and log_date > date.today():
            raise forms.ValidationError("Date cannot be in the future.")

        if entry and exit_ and entry == exit_:
            raise forms.ValidationError(
                "Entry and exit time cannot be the same."
            )

        return cleaned_data