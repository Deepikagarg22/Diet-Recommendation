from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import (
    UserCreationForm, AuthenticationForm
)
from .models import UserProfile


class UserRegistrationForm(UserCreationForm):
    """Registration form with profile fields included."""

    # ── User fields ──
    first_name = forms.CharField(
        max_length=50,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'First Name',
            'autofocus': True,
        })
    )
    last_name = forms.CharField(
        max_length=50,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Last Name',
        })
    )
    username = forms.CharField(
        max_length=50,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Username',
        })
    )
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Email Address',
        })
    )
    password1 = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Password',
        })
    )
    password2 = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Confirm Password',
        })
    )

    # ── Profile fields ──
    age = forms.IntegerField(
        min_value=10, max_value=100,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Age',
        })
    )
    gender = forms.ChoiceField(
        choices=UserProfile.GENDER_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-control',
        })
    )
    weight_kg = forms.FloatField(
        min_value=20, max_value=300,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Weight (kg)',
            'step': '0.1',
        })
    )
    height_cm = forms.FloatField(
        min_value=100, max_value=250,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Height (cm)',
            'step': '0.1',
        })
    )
    activity_level = forms.ChoiceField(
        choices=UserProfile.ACTIVITY_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-control',
        })
    )
    purpose = forms.ChoiceField(
        choices=UserProfile.PURPOSE_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-control',
        })
    )
    diet_preference = forms.ChoiceField(
        choices=UserProfile.DIET_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-control',
        })
    )

    class Meta:
        model = User
        fields = [
            'first_name', 'last_name', 'username',
            'email', 'password1', 'password2',
        ]

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError(
                "A user with this email already exists."
            )
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        if commit:
            user.save()
            # Update the auto-created profile
            profile = user.profile
            profile.age = self.cleaned_data['age']
            profile.gender = self.cleaned_data['gender']
            profile.weight_kg = self.cleaned_data['weight_kg']
            profile.height_cm = self.cleaned_data['height_cm']
            profile.activity_level = self.cleaned_data['activity_level']
            profile.purpose = self.cleaned_data['purpose']
            profile.diet_preference = self.cleaned_data['diet_preference']
            profile.save()
        return user


class UserLoginForm(AuthenticationForm):
    """Styled login form."""

    username = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Username',
            'autofocus': True,
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Password',
        })
    )


class UserUpdateForm(forms.ModelForm):
    """Form to update User model fields."""

    first_name = forms.CharField(
        max_length=50,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
        })
    )
    last_name = forms.CharField(
        max_length=50,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
        })
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
        })
    )

    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']

    def clean_email(self):
        email = self.cleaned_data.get('email')
        qs = User.objects.filter(email=email).exclude(
            pk=self.instance.pk
        )
        if qs.exists():
            raise forms.ValidationError(
                "A user with this email already exists."
            )
        return email


class ProfileUpdateForm(forms.ModelForm):
    """Form to update UserProfile model fields."""

    class Meta:
        model = UserProfile
        fields = [
            'age', 'gender', 'weight_kg', 'height_cm',
            'activity_level', 'purpose', 'diet_preference',
        ]
        widgets = {
            'age': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 10, 'max': 100,
            }),
            'gender': forms.Select(attrs={
                'class': 'form-control',
            }),
            'weight_kg': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.1', 'min': 20, 'max': 300,
            }),
            'height_cm': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.1', 'min': 100, 'max': 250,
            }),
            'activity_level': forms.Select(attrs={
                'class': 'form-control',
            }),
            'purpose': forms.Select(attrs={
                'class': 'form-control',
            }),
            'diet_preference': forms.Select(attrs={
                'class': 'form-control',
            }),
        }