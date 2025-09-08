const Brevo = require('@getbrevo/brevo');
require('dotenv').config();

const brevoClient = new Brevo.TransactionalEmailsApi();
brevoClient.setApiKey(Brevo.TransactionalEmailsApiApiKeys.apiKey, process.env.BREVO_API_KEY);

const FROM_EMAIL = 'no-reply@sipbrewery.com';
const FROM_NAME = 'SIP Brewery';

function sendVerificationEmail(email, name, token) {
  const verificationUrl = `https://sipbrewery.com/verify-email?token=${token}`;
  const subject = 'Verify Your SIP Brewery Account';
  const htmlContent = `
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:auto;padding:24px;background:#f9f9f9;border-radius:8px;">
      <h2 style="color:#2d7ff9;">Welcome to SIP Brewery, ${name}!</h2>
      <p>Thank you for registering. Please verify your email address by clicking the button below:</p>
      <a href="${verificationUrl}" style="display:inline-block;padding:12px 24px;background:#2d7ff9;color:#fff;text-decoration:none;border-radius:4px;font-weight:bold;">Verify Email</a>
      <p style="margin-top:24px;font-size:12px;color:#888;">If you did not create an account, you can ignore this email.</p>
    </div>
  `;
  const sendSmtpEmail = {
    to: [{ email, name }],
    sender: { email: FROM_EMAIL, name: FROM_NAME },
    subject,
    htmlContent
  };
  return brevoClient.sendTransacEmail(sendSmtpEmail);
}

function sendPasswordResetEmail(email, name, token) {
  const resetUrl = `https://sipbrewery.com/reset-password?token=${token}`;
  const subject = 'Reset Your SIP Brewery Password';
  const htmlContent = `
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:auto;padding:24px;background:#f9f9f9;border-radius:8px;">
      <h2 style="color:#2d7ff9;">Password Reset Request</h2>
      <p>Hello ${name},</p>
      <p>We received a request to reset your SIP Brewery password. Click the button below to set a new password:</p>
      <a href="${resetUrl}" style="display:inline-block;padding:12px 24px;background:#2d7ff9;color:#fff;text-decoration:none;border-radius:4px;font-weight:bold;">Reset Password</a>
      <p style="margin-top:24px;font-size:12px;color:#888;">If you did not request a password reset, you can ignore this email.</p>
    </div>
  `;
  const sendSmtpEmail = {
    to: [{ email, name }],
    sender: { email: FROM_EMAIL, name: FROM_NAME },
    subject,
    htmlContent
  };
  return brevoClient.sendTransacEmail(sendSmtpEmail);
}

module.exports = {
  sendVerificationEmail,
  sendPasswordResetEmail
}; 